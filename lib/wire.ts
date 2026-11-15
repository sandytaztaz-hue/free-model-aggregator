// lib/wire.ts — the client wires and the three upstream wires.
//
// The entry serves two client APIs (OpenAI Chat Completions, Anthropic Messages),
// each in JSON and in server-sent events, over providers that speak three wires
// (openai, anthropic, gemini). This file translates in both directions:
//
//   client request  -> canonical form -> upstream request on the provider's wire
//   upstream answer -> canonical form -> client answer on the client's wire
//
// The canonical form keeps exactly what the providers echo back, so a translation
// never drops a field the client sent: the system prompt, the user text (images
// included), the tool names, the output cap, the stop sequences and JSON mode.
//
// Streaming event order follows each wire's own shape:
//   OpenAI:    role chunk, content deltas, finish chunk, usage chunk, [DONE]
//   Anthropic: message_start, content_block_start, deltas, content_block_stop,
//              message_delta, message_stop
//   Gemini:    one frame per text piece, a final frame with finishReason + usage
//
// Everything here is a pure function of its arguments: no clock and no randomness.
// Model lists use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The Anthropic API version the entry sends upstream. */
export const ANTHROPIC_VERSION = '2023-06-01';

// ---------------------------------------------------------------------------
// Model lists and error envelopes
// ---------------------------------------------------------------------------

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Canonical request form
// ---------------------------------------------------------------------------

export type CanonicalPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string };

export interface CanonicalToolCall {
  id: string;
  name: string;
  args: Json;
}

export interface CanonicalMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: CanonicalPart[];
  toolCalls: CanonicalToolCall[];
  toolCallId: string;
}

export interface CanonicalTool {
  name: string;
  description: string;
  parameters: Json;
}

export type CanonicalToolChoice = 'auto' | 'none' | 'any' | { name: string } | null;

export interface CanonicalRequest {
  system: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  toolChoice: CanonicalToolChoice;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  stream: boolean;
  jsonMode: boolean;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as a canonical part: text or an inline image; null for the rest. */
function anthropicPartOf(block: Json): CanonicalPart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { kind: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { kind: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { kind: 'imageUrl', url: source.url };
  }
  return null;
}

/** One OpenAI content part as a canonical part: text or an image URL; null for the rest. */
function openAIPartOf(block: Json): CanonicalPart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { kind: 'text', text: block.text };
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const url = block.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { kind: 'image', mediaType: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
    return { kind: 'imageUrl', url };
  }
  if (block.type === 'image' && typeof block.url === 'string') return { kind: 'imageUrl', url: block.url };
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions request as the canonical form. */
export function openAIToCanonical(body: Json): CanonicalRequest {
  const messages: CanonicalMessage[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: CanonicalToolCall[] = [];
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: stringOf(call.id),
          name: stringOf(call.function.name),
          args: argumentsOf(call.function.arguments),
        });
      }
      const parts: CanonicalPart[] = [];
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') parts.push({ kind: 'text', text });
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: '' });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        parts: [{ kind: 'text', text: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
        toolCalls: [],
        toolCallId: stringOf(raw.tool_call_id),
      });
      continue;
    }
    // Any other role is a user turn.
    const parts: CanonicalPart[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const part = openAIPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    messages.push({ role: 'user', parts, toolCalls: [], toolCallId: '' });
  }
  const tools: CanonicalTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: stringOf(raw.function.description),
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  let toolChoice: CanonicalToolChoice = null;
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none' || choice === 'required') toolChoice = choice === 'required' ? 'any' : choice;
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops,
    stream: body.stream === true,
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** An Anthropic Messages request as the canonical form. */
export function anthropicToCanonical(body: Json): CanonicalRequest {
  const system = textOf(body.system);
  const messages: CanonicalMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: CanonicalPart[] = [];
      const toolCalls: CanonicalToolCall[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_use') {
            toolCalls.push({
              id: stringOf(block.id),
              name: stringOf(block.name),
              args: isRecord(block.input) ? block.input : {},
            });
          } else {
            const part = anthropicPartOf(block);
            if (part !== null) parts.push(part);
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') {
        parts.push({ kind: 'text', text: raw.content });
      }
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: '' });
      continue;
    }
    // A user turn: text and image parts stay in the turn; each tool_result becomes its own tool message.
    const parts: CanonicalPart[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          messages.push({
            role: 'tool',
            parts: [{ kind: 'text', text: toolResultText(block.content) }],
            toolCalls: [],
            toolCallId: stringOf(block.tool_use_id),
          });
          continue;
        }
        const part = anthropicPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', parts, toolCalls: [], toolCallId: '' });
  }
  const tools: CanonicalTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: stringOf(raw.description),
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let toolChoice: CanonicalToolChoice = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'any';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) for (const s of body.stop_sequences) if (typeof s === 'string') stops.push(s);
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops,
    stream: body.stream === true,
    jsonMode: false,
  };
}

// ---------------------------------------------------------------------------
// Canonical form -> upstream request bodies
// ---------------------------------------------------------------------------

function canonicalText(parts: readonly CanonicalPart[]): string {
  return parts.flatMap((p) => (p.kind === 'text' ? [p.text] : [])).join('\n');
}

/** The canonical form as an OpenAI Chat Completions request naming `model`. */
export function canonicalToOpenAI(c: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (c.system !== '') messages.push({ role: 'system', content: c.system });
  for (const message of c.messages) {
    if (message.role === 'assistant') {
      const out: Json = { role: 'assistant', content: canonicalText(message.parts) !== '' ? canonicalText(message.parts) : null };
      if (message.toolCalls.length > 0) {
        out.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: JSON.stringify(call.args) },
        }));
      }
      messages.push(out);
      continue;
    }
    if (message.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: message.toolCallId, content: canonicalText(message.parts) });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') parts.push({ type: 'image_url', image_url: { url: `data:${part.mediaType};base64,${part.data}` } });
      else parts.push({ type: 'image_url', image_url: { url: part.url } });
    }
    messages.push({ role: 'user', content: parts.length === 1 && parts[0]?.type === 'text' ? (parts[0] as Json).text : parts });
  }
  const out: Json = { model, messages };
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop = c.stops;
  if (c.tools.length > 0) {
    out.tools = c.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  if (c.toolChoice !== null) {
    if (c.toolChoice === 'auto') out.tool_choice = 'auto';
    else if (c.toolChoice === 'any') out.tool_choice = 'required';
    else if (c.toolChoice === 'none') out.tool_choice = 'none';
    else out.tool_choice = { type: 'function', function: { name: c.toolChoice.name } };
  }
  if (c.jsonMode) out.response_format = { type: 'json_object' };
  if (c.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical form as an Anthropic Messages request naming `model`. */
export function canonicalToAnthropic(c: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of c.messages) {
    if (message.role === 'assistant') {
      const content: Json[] = [];
      for (const part of message.parts) if (part.kind === 'text') content.push({ type: 'text', text: part.text });
      for (const call of message.toolCalls) {
        content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (message.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: message.toolCallId, content: canonicalText(message.parts) }],
      });
      continue;
    }
    const content: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') content.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') content.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } });
      else content.push({ type: 'image', source: { type: 'url', url: part.url } });
    }
    messages.push({ role: 'user', content });
  }
  const out: Json = { model, messages };
  if (c.system !== '') out.system = c.system;
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop_sequences = c.stops;
  if (c.tools.length > 0) {
    out.tools = c.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (c.toolChoice !== null) {
    if (c.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
    else if (c.toolChoice === 'any') out.tool_choice = { type: 'any' };
    else if (c.toolChoice === 'none') out.tool_choice = { type: 'none' };
    else out.tool_choice = { type: 'tool', name: c.toolChoice.name };
  }
  if (c.stream) out.stream = true;
  return out;
}

/** One canonical part as a Gemini content part. */
function geminiPartOf(part: CanonicalPart): Json | null {
  if (part.kind === 'text') return { text: part.text };
  if (part.kind === 'image') return { inlineData: { mimeType: part.mediaType, data: part.data } };
  return { fileData: { fileUri: part.url } };
}

/** The canonical form as a Gemini generateContent request body (the model goes in the path). */
export function canonicalToGemini(c: CanonicalRequest): Json {
  const contents: Json[] = [];
  for (const message of c.messages) {
    if (message.role === 'assistant') {
      const parts: Json[] = [];
      for (const part of message.parts) if (part.kind === 'text') parts.push({ text: part.text });
      for (const call of message.toolCalls) parts.push({ functionCall: { name: call.name, args: call.args } });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (message.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { content: canonicalText(message.parts) } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      const geminiPart = geminiPartOf(part);
      if (geminiPart !== null) parts.push(geminiPart);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (c.system !== '') out.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: c.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.temperature !== null) config.temperature = c.temperature;
  if (c.topP !== null) config.topP = c.topP;
  if (c.stops.length > 0) config.stopSequences = c.stops;
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Stop reasons and usage
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

interface Usage {
  input: number;
  output: number;
}

function usageOfOpenAI(body: Json): Usage {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) };
}

function usageOfAnthropic(body: Json): Usage {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) };
}

function usageOfGemini(body: Json): Usage {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

// ---------------------------------------------------------------------------
// Upstream JSON answers -> client JSON answers
// ---------------------------------------------------------------------------

interface ParsedAnswer {
  text: string;
  toolCalls: CanonicalToolCall[];
  stopFamily: 'stop' | 'length' | 'tool' | 'refusal';
  usage: Usage;
}

function parseOpenAIAnswer(body: Json): ParsedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: CanonicalToolCall[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({ id: stringOf(call.id), name: stringOf(call.function.name), args: argumentsOf(call.function.arguments) });
  }
  const finish = choice.finish_reason;
  return {
    text,
    toolCalls,
    stopFamily: finish === 'length' ? 'length' : finish === 'tool_calls' || finish === 'function_call' ? 'tool' : finish === 'content_filter' ? 'refusal' : 'stop',
    usage: usageOfOpenAI(body),
  };
}

function parseAnthropicAnswer(body: Json): ParsedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const toolCalls: CanonicalToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({ id: stringOf(block.id), name: stringOf(block.name), args: isRecord(block.input) ? block.input : {} });
  }
  const reason = body.stop_reason;
  return {
    text,
    toolCalls,
    stopFamily: reason === 'max_tokens' ? 'length' : reason === 'tool_use' ? 'tool' : reason === 'refusal' ? 'refusal' : 'stop',
    usage: usageOfAnthropic(body),
  };
}

function parseGeminiAnswer(body: Json): ParsedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const toolCalls: CanonicalToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part) || !isRecord(part.functionCall)) continue;
    const call = part.functionCall;
    toolCalls.push({ id: `call_${stringOf(call.name)}`, name: stringOf(call.name), args: isRecord(call.args) ? call.args : {} });
  }
  const reason = candidate.finishReason;
  return {
    text,
    toolCalls,
    stopFamily: reason === 'MAX_TOKENS' ? 'length' : reason === 'SAFETY' ? 'refusal' : 'stop',
    usage: usageOfGemini(body),
  };
}

/** A parsed upstream answer as an OpenAI Chat Completions object naming `model`. */
export function answerToOpenAI(parsed: ParsedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: parsed.text !== '' || parsed.toolCalls.length === 0 ? parsed.text : null };
  if (parsed.toolCalls.length > 0) {
    message.tool_calls = parsed.toolCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }
  const finish = parsed.stopFamily === 'length' ? 'length' : parsed.stopFamily === 'tool' ? 'tool_calls' : parsed.stopFamily === 'refusal' ? 'content_filter' : 'stop';
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: { prompt_tokens: parsed.usage.input, completion_tokens: parsed.usage.output, total_tokens: parsed.usage.input + parsed.usage.output },
  };
}

/** A parsed upstream answer as an Anthropic Messages object naming `model`. */
export function answerToAnthropic(parsed: ParsedAnswer, model: string): Json {
  const content: Json[] = [];
  if (parsed.text !== '') content.push({ type: 'text', text: parsed.text });
  for (const call of parsed.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const stop = parsed.stopFamily === 'length' ? 'max_tokens' : parsed.stopFamily === 'tool' ? 'tool_use' : parsed.stopFamily === 'refusal' ? 'refusal' : 'end_turn';
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: parsed.usage.input, output_tokens: parsed.usage.output },
  };
}

/** An upstream JSON body on `wire` as an OpenAI Chat Completions object naming `model`. */
export function upstreamJsonToOpenAI(wire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string): Json {
  if (wire === 'openai') return body;
  const parsed = wire === 'anthropic' ? parseAnthropicAnswer(body) : parseGeminiAnswer(body);
  return answerToOpenAI(parsed, model);
}

/** An upstream JSON body on `wire` as an Anthropic Messages object naming `model`. */
export function upstreamJsonToAnthropic(wire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string): Json {
  if (wire === 'anthropic') return body;
  const parsed = wire === 'openai' ? parseOpenAIAnswer(body) : parseGeminiAnswer(body);
  return answerToAnthropic(parsed, model);
}

// ---------------------------------------------------------------------------
// SSE plumbing
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames: each frame's event name (or null) and its joined data lines. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Yield the SSE frames of a byte stream as they arrive, carrying a partial frame across reads. */
export async function* sseFramesOf(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const parsed of parseSseFrames(frame)) yield parsed;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const parsed of parseSseFrames(buffer)) yield parsed;
  } finally {
    reader.releaseLock();
  }
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// OpenAI SSE <-> Anthropic SSE
// ---------------------------------------------------------------------------

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the
 * first chunk, content_block_start/delta/stop per text block or tool call, then
 * message_delta with the stop reason and usage, then message_stop. `model` is the
 * model the client asked for, named back to it.
 */
export async function* openAIStreamToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = function* (): Generator<string> {
    if (started) return;
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    });
  };
  const closeEvents = function* (): Generator<string> {
    if (textIndex !== -1) {
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    toolIndexes.clear();
  };
  const finishEvents = function* (stopReason: string): Generator<string> {
    if (finished) return;
    finished = true;
    yield* closeEvents();
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  };

  for await (const frame of sseFramesOf(stream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
    // OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openAIStopToAnthropic(choice.finish_reason);
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/**
 * An Anthropic SSE stream as Chat Completions chunks: a role chunk on message_start,
 * content deltas for text and tool calls, a finish chunk on message_delta, a usage
 * chunk when both token counts are known, then [DONE].
 */
export async function* anthropicStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let sentRole = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let sawUsage = false;
  const toolByBlock = new Map<number, { id: string; name: string; index: number }>();
  let nextToolIndex = 0;

  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });

  for await (const frame of sseFramesOf(stream)) {
    if (frame.event === null) continue;
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      if (!sentRole) {
        sentRole = true;
        yield chunk({ role: 'assistant', content: '' }, null);
      }
      continue;
    }
    if (frame.event === 'content_block_start') {
      if (!sentRole) {
        sentRole = true;
        yield chunk({ role: 'assistant', content: '' }, null);
      }
      const index = numberOrNull(data.index);
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (index !== null && block.type === 'tool_use') {
        toolByBlock.set(index, {
          id: typeof block.id === 'string' && block.id !== '' ? block.id : `call_${index}`,
          name: typeof block.name === 'string' ? block.name : '',
          index: nextToolIndex,
        });
        nextToolIndex += 1;
      }
      continue;
    }
    if (frame.event === 'content_block_delta') {
      if (!sentRole) {
        sentRole = true;
        yield chunk({ role: 'assistant', content: '' }, null);
      }
      const index = numberOrNull(data.index);
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield chunk({ content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '' && index !== null) {
        const tool = toolByBlock.get(index);
        if (tool !== undefined) {
          yield chunk({ tool_calls: [{ index: tool.index, id: tool.id, type: 'function', function: { name: tool.name, arguments: delta.partial_json } }] }, null);
        }
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (isRecord(data.usage)) {
        outputTokens = numberOrZero(data.usage.output_tokens);
        sawUsage = true;
      }
      if (typeof delta.stop_reason === 'string') {
        finished = true;
        yield chunk({}, anthropicStopToOpenAI(delta.stop_reason));
      }
      continue;
    }
    if (frame.event === 'message_stop') {
      if (!finished) {
        finished = true;
        yield chunk({}, 'stop');
      }
      if (sawUsage) {
        yield sseEvent(null, {
          id: 'chatcmpl-agg',
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [],
          usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
        });
      }
      yield 'data: [DONE]\n\n';
      return;
    }
  }
  if (!finished) yield chunk({}, 'stop');
  yield 'data: [DONE]\n\n';
}

// ---------------------------------------------------------------------------
// Gemini SSE <-> OpenAI / Anthropic SSE
// ---------------------------------------------------------------------------

interface GeminiFrame {
  text: string;
  functionCalls: { name: string; args: Json }[];
  finishReason: string | null;
  usage: Usage | null;
}

function parseGeminiFrame(data: Json): GeminiFrame {
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const functionCalls: { name: string; args: Json }[] = [];
  for (const part of parts) {
    if (!isRecord(part) || !isRecord(part.functionCall)) continue;
    const call = part.functionCall;
    functionCalls.push({ name: stringOf(call.name), args: isRecord(call.args) ? call.args : {} });
  }
  const usage = isRecord(data.usageMetadata)
    ? { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) }
    : null;
  return { text, functionCalls, finishReason: typeof candidate.finishReason === 'string' ? candidate.finishReason : null, usage };
}

/** A Gemini streamGenerateContent SSE stream as Chat Completions chunks, then [DONE]. */
export async function* geminiStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let sentRole = false;
  let finished = false;
  let usage: Usage | null = null;
  let toolIndex = 0;
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });

  for await (const frame of sseFramesOf(stream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    const parsed = parseGeminiFrame(data);
    if (parsed.usage !== null) usage = parsed.usage;
    if (!sentRole) {
      sentRole = true;
      yield chunk({ role: 'assistant', content: '' }, null);
    }
    if (parsed.text !== '') yield chunk({ content: parsed.text }, null);
    for (const call of parsed.functionCalls) {
      yield chunk({ tool_calls: [{ index: toolIndex, id: `call_${call.name}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }] }, null);
      toolIndex += 1;
    }
    if (parsed.finishReason !== null) {
      finished = true;
      yield chunk({}, geminiStopToOpenAI(parsed.finishReason));
    }
  }
  if (!finished) yield chunk({}, 'stop');
  if (usage !== null) {
    yield sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    });
  }
  yield 'data: [DONE]\n\n';
}

/** A Gemini streamGenerateContent SSE stream as the Anthropic event sequence. */
export async function* geminiStreamToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textOpen = false;
  let nextIndex = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';

  for await (const frame of sseFramesOf(stream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    const parsed = parseGeminiFrame(data);
    if (!started) {
      started = true;
      yield sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: parsed.usage?.input ?? 0, output_tokens: 0 } },
      });
    }
    if (parsed.text !== '') {
      if (!textOpen) {
        textOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: nextIndex, content_block: { type: 'text', text: '' } });
        nextIndex += 1;
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: nextIndex - 1, delta: { type: 'text_delta', text: parsed.text } });
    }
    for (const call of parsed.functionCalls) {
      if (textOpen) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex - 1 });
        textOpen = false;
      }
      const index = nextIndex;
      nextIndex += 1;
      yield sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: `toolu_${call.name}`, name: call.name, input: {} },
      });
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.args) } });
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
      stopReason = 'tool_use';
    }
    if (parsed.finishReason !== null) {
      if (parsed.finishReason !== 'STOP') stopReason = geminiStopToAnthropic(parsed.finishReason);
      if (parsed.usage !== null) outputTokens = parsed.usage.output;
      if (textOpen) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex - 1 });
        textOpen = false;
      }
      finished = true;
      yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
      yield sseEvent('message_stop', { type: 'message_stop' });
    } else if (parsed.usage !== null) {
      outputTokens = parsed.usage.output;
    }
  }
  if (!finished) {
    if (!started) {
      yield sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
    }
    if (textOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex - 1 });
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}

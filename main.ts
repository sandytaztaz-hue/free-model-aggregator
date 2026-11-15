// main.ts — one server, two client wires, a directory of free providers.
//
// The trial gives this server a brief and an env, and runs it as `node main.ts`
// from the repository root, with Node 22's standard library alone (TypeScript
// runs directly; a relative import needs its .ts extension). The brief is at the
// path in $MAKEFUN_PROMPT_FILE and carries three JSON documents:
//   - the provider directory: ten simulated free-tier providers, each with its id,
//     wire (openai, anthropic or gemini), auth scheme, key environment variable,
//     the key the service takes, and the base path to join to $MAKEFUN_ENV_URL;
//   - the quality table: every model's public rating and what it can do (tools,
//     vision, JSON mode, context length), and whether it is free;
//   - the workload manifest: the twenty requests the client will send, in order,
//     with each one's api, whether it streams, its weight and the requirements a
//     model must meet to serve it.
// The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths.
// This server reads the brief, probes each provider's catalog route in the
// provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// Routing: every request's free candidates are ranked by the sealed quality table's
// rating_milli (see lib/router.ts), planned against each provider's risk-aware
// per-trial call cap with the scarce providers reserved for the requests that need
// them most, and tried in that order. A provider that refuses (402, 429, 5xx, or
// unreachable) is never retried for that request — a retry would call the same
// provider's next call number, which is as likely to refuse, and only spends
// another upstream call — the request moves straight to the next-best free model
// that meets it. A provider error is never forwarded to the client while an
// untried candidate remains.
//
// Every choice is a function of the brief, the requests and the providers'
// answers: no clock, no unseeded randomness, so a replay leaves the same state
// export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToCanonical,
  canonicalToAnthropic,
  canonicalToGemini,
  canonicalToOpenAI,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIToCanonical,
  upstreamJsonToAnthropic,
  upstreamJsonToOpenAI,
  ANTHROPIC_VERSION,
  type CanonicalRequest,
  type ServedModel,
} from './lib/wire.ts';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  attemptOrder,
  candidatesFor,
  orderAttempts,
  planPrimaries,
  RouterState,
  type Candidate,
} from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The router's live state: calls made per provider, and which providers look dead. */
const router = new RouterState();
/** Consecutive malformed 2xx answers per provider: two in a row mark it dead, like any other failure. */
const malformedStreaks = new Map<string, number>();

/**
 * Record a 2xx answer that is not a well-formed answer of its wire. It already
 * counted as a successful call against the provider's budget (callUpstream noted
 * it so), so this only moves the failure streak: two malformed answers in a row
 * mark the provider dead for the rest of the trial, instead of leaving a
 * provider that only ever answers garbage looking healthy forever.
 */
function noteMalformed(providerId: string): void {
  const streak = (malformedStreaks.get(providerId) ?? 0) + 1;
  malformedStreaks.set(providerId, streak);
  if (streak >= 2) router.dead.add(providerId);
}
/** Each manifest request's planned primary candidate, assigned at startup. */
let primaries: (Candidate | null)[] = [];

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog, and plan the workload's primaries. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  primaries = planPrimaries(probes, brief.ratings, brief.workload?.manifest ?? []);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type UpstreamResult = { ok: true; status: number; text: string } | { ok: false; status: number };

/**
 * One inference call to one candidate's provider, in that provider's own wire.
 * The whole answer is read before it returns, so a caller only ever answers the
 * client from a complete upstream answer — a stream that breaks mid-way is a
 * failed attempt, not a partial answer. A call that names a free model counts
 * against the provider's budget even when it is refused, so every outcome is
 * recorded against the router's state.
 */
async function callUpstream(candidate: Candidate, canonical: CanonicalRequest, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = canonicalToOpenAI(canonical, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    body = canonicalToAnthropic(canonical, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${stream ? ':streamGenerateContent?alt=sse' : ':generateContent'}`;
    body = canonicalToGemini(canonical);
  }
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    router.noteCall(provider.id, false);
    return { ok: false, status: 0 };
  }
  const text = await res.text().catch(() => '');
  if (!res.ok) {
    router.noteCall(provider.id, false);
    return { ok: false, status: res.status };
  }
  router.noteCall(provider.id, true);
  return { ok: true, status: res.status, text };
}

/** One SSE text body as a byte stream, for the stream converters. */
function streamOfText(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

/**
 * A complete upstream answer as the client's answer. Same-wire streams pass
 * through byte for byte; cross-wire answers are translated, JSON and streaming.
 * Returns false when the upstream text is not a well-formed answer of its wire,
 * so the caller can fail over instead of answering with it.
 */
async function answerClient(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstreamText: string,
  stream: boolean,
): Promise<boolean> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (stream) {
    if (clientApi === 'openai' && wire === 'openai') {
      sendText(res, 200, 'text/event-stream', upstreamText);
      return true;
    }
    if (clientApi === 'anthropic' && wire === 'anthropic') {
      sendText(res, 200, 'text/event-stream', upstreamText);
      return true;
    }
    const source = streamOfText(upstreamText);
    const events: string[] = [];
    try {
      if (clientApi === 'openai') {
        const gen = wire === 'anthropic' ? anthropicStreamToOpenAI(source, model) : geminiStreamToOpenAI(source, model);
        for await (const event of gen) events.push(event);
      } else {
        const gen = wire === 'openai' ? openAIStreamToAnthropic(source, model) : geminiStreamToAnthropic(source, model);
        for await (const event of gen) events.push(event);
      }
    } catch {
      return false;
    }
    sendText(res, 200, 'text/event-stream', events.join(''));
    return true;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(upstreamText);
  } catch {
    return false;
  }
  if (!isRecord(parsed)) return false;
  const out = clientApi === 'openai' ? upstreamJsonToOpenAI(wire, parsed, model) : upstreamJsonToAnthropic(wire, parsed, model);
  sendJson(res, 200, out);
  return true;
}

/**
 * Serve one client request: try the ranked candidates in order, with one
 * same-provider retry while the provider is still inside its conservative
 * budget, until one serves or the candidates run out. A provider error is never
 * forwarded while an untried candidate remains.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body, clientApi);
  const canonical = clientApi === 'openai' ? openAIToCanonical(body) : anthropicToCanonical(body);
  const stream = canonical.stream;
  const index = manifestIndex - 1;
  const candidates = candidatesFor(probes, brief.ratings, requires, clientApi);
  const primary = index >= 0 && index < primaries.length ? primaries[index] : null;
  const ordered = orderAttempts(attemptOrder(primary, candidates), router);
  for (const candidate of ordered) {
    const result = await callUpstream(candidate, canonical, stream);
    if (result.ok) {
      if (await answerClient(res, clientApi, candidate, result.text, stream)) return;
      // A 2xx that is not a well-formed answer of its wire: treat it as a failed attempt,
      // and count it toward marking the provider dead so garbage answers fail over fast.
      noteMalformed(candidate.providerId);
      continue;
    }
    // A refusal: never retry the same provider for this request. A refusal
    // here is a scheduled failure, a spent budget or an outage, and the same
    // provider's next call number is as likely to refuse as this one was, so
    // a retry only spends another upstream call; move straight to the next
    // candidate instead. (An earlier version retried once: on the practice
    // trials the retry itself refused whenever the next call number was also
    // on the provider's failure schedule, turning one extra call into two.)
    continue;
  }
  sendError(
    res,
    clientApi,
    503,
    clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'no free model on the directory could serve this request',
  );
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider's wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider's wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await sleep(200);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

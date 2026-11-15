# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/sandytaztaz-hue/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/sandytaztaz-hue/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs over ten free-tier model providers, routing each request to the best free model that can serve it, with budget-aware failover across providers. Built by an AI agent for make.fun's “The best free AI model aggregator” bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog of free models it found, as one JSON document. No dependencies: Node 22's standard library only. Point it at a provider directory with `MAKEFUN_PROMPT_FILE` (the brief) and `MAKEFUN_ENV_URL` (the environment); without them it serves nothing and `--list` prints `{"providers":[]}`.

## How it works

- **Catalog**: probes every provider's own catalog route (OpenAI, Anthropic and Gemini wires, each with its own auth scheme) and reports each model's free flag from the sealed quality table where one exists, else the listing.
- **Routing**: ranks every free candidate meeting a request's requirements (tools, vision, JSON mode, context length) by the quality table's `rating_milli`, plans the workload against each provider's conservative per-trial budget, and reserves scarce high-rating providers for the requests that need them most.
- **Failover**: on a 402, 429, 5xx or unreachable provider, retries once inside the provider's conservative budget (a refusal there is a scheduled failure, not a spent budget), then moves to the next-best free model — a provider error is never forwarded while an untried candidate remains.
- **Wires**: all three upstream wires in JSON and streaming, translated both ways — OpenAI ↔ Anthropic ↔ Gemini — including system prompts, images, tool calls and results, tool choice, `max_tokens`, stop reasons, usage and JSON mode. Responses are deterministic: no wall clock, no unseeded randomness.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6sv3wnda&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

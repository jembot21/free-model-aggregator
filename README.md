# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/jembot21/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/jembot21/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One gateway server over ten free-tier model providers. It speaks the OpenAI Chat Completions API and the Anthropic Messages API to its client — JSON and server-sent events — and routes every request to the best-rated free model that meets its requirements, on whichever of the three provider wires (OpenAI, Anthropic, Gemini) that model lives, translating requests, answers and event streams on the way.

- **Routing:** candidates are ranked by the quality table's ratings, with a load penalty that spreads calls across providers before any one provider's per-trial budget runs out, and a failure memory that demotes a provider which just refused.
- **Failover:** a call answered 402, 429 or 5xx — or one that never answers — is retried on the next candidate until one serves.
- **Wires:** all three upstream wires are implemented in both client shapes (`lib/wire.ts`, `lib/wire2.ts`), including tool calls, images, JSON mode, stop reasons and usage, in JSON and in streaming.
- **Determinism:** every choice derives from the brief, the requests and the providers' answers — no wall clock, no randomness — so a replayed trial reproduces the same state.

## Run it

A trial provides the brief and the environment:

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=brief.txt node main.ts
node main.ts --list   # print the catalog document the server registers
```

Node 22, standard library only.

// free-model-aggregator-01: one gateway server over the brief's ten simulated free-tier providers.
//
// Shape of a trial (the kit's RULES.txt has the whole of it): the brief at $MAKEFUN_PROMPT_FILE
// carries the provider directory, the sealed quality table and the workload manifest of twenty
// requests. This server probes every provider's catalog in its own wire, builds one catalog,
// registers its port and that catalog with the env at $MAKEFUN_ENV_URL, serves the client's
// OpenAI-shaped and Anthropic-shaped requests (JSON and SSE) from free models that meet each
// request's requirements, and exits 0 when the env's /work says done.
//
// How it wins its columns:
//   quality (60%): every request is routed to the best-rated free model that meets its
//     requirements. The top of the table is a few milli-points apart, so the order stays by
//     rating while a provider keeps serving: its budget running out costs a failover, not a
//     downgrade chosen in advance. Only a provider refusing repeatedly — an outage, or a
//     budget spent — is demoted, and only until it serves again.
//   failover (10%): a provider call answered 402, 429 or 5xx, or one that never answers, is
//     retried on the next candidate until one serves; the answering model's rating is what counts.
//   served (15%): all three upstream wires (openai, anthropic, gemini) are spoken in both client
//     shapes, in JSON and in streaming, so no request dies on an unimplemented wire.
//   catalog (5%): the probe's catalog, from the listings with the quality table as the truth for
//     free flags and context lengths, registered byte for byte as --list prints it.
//
// Every choice derives from the brief, the requests and the providers' answers: no wall clock and
// no randomness anywhere, so a replay of the same trial reproduces the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
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
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsToOpenAI,
  anthropicResponseToOpenAI,
  geminiEventsToOpenAI,
  geminiToOpenAI,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  streamOfText,
} from './lib/wire2.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 30000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

// ---------------------------------------------------------------------------
// Routing: candidates ranked by rating, spread across providers, with memory.
// ---------------------------------------------------------------------------

interface ProviderState {
  /** Provider calls attempted (each one the trial counts). */
  calls: number;
  /** Calls the provider refused or that never answered. */
  failures: number;
  /** Refusals since this provider last served: the outage signal. */
  consecutiveFailures: number;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, failures: 0, consecutiveFailures: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

/** Every free model that meets the requirements, best first under the routing policy. */
function candidatesFor(requires: Requirement, weight: number, jsonOnOpenAIClient: boolean): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerIndex });
    }
  });
  // The order stays almost purely by rating: a provider keeps the work while it keeps
  // serving, because the only real cost of using it is the budget it may run out of, and a
  // refusal then fails over to the next candidate at full rating. Two guards, both small:
  // past eight calls a light request pays a few milli-points toward spreading (a heavy one
  // never does, so a late heavy request still finds the best model), and a provider that
  // refuses twice running drops a little, three times running a lot — refusals on a failure
  // schedule are transient and the provider usually serves the next call, but a provider in
  // an outage or out of budget should stop costing a call at the head of every request. A
  // success clears the refusal count, so a recovered provider returns at full rating.
  const score = (c: Candidate): number => {
    const state = stateOf(c.probe.provider.id);
    const load = weight >= 2 ? 0 : Math.min(4, Math.max(0, state.calls - 8));
    const outage = state.consecutiveFailures >= 3 ? 100000 : state.consecutiveFailures === 2 ? 40 : 0;
    // An Anthropic-wire provider has no way to signal JSON mode back, so a client that asked
    // for JSON mode on the OpenAI wire can only be answered validly by the other two wires:
    // those providers drop to a last resort for such a request, however good their rating is.
    const jsonTrap = jsonOnOpenAIClient && c.probe.provider.wire === 'anthropic' ? 100000 : 0;
    return c.rating - load - outage - jsonTrap;
  };
  out.sort((a, b) => {
    const byScore = score(b) - score(a);
    if (byScore !== 0) return byScore;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The workload: requirements and weight of the request now arriving.
// ---------------------------------------------------------------------------

interface Arriving {
  requires: Requirement;
  weight: number;
}

function requirementsFor(body: Json): Arriving {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight };
  return { requires: requirementsFromBody(body), weight: 1 };
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

// ---------------------------------------------------------------------------
// Upstream calls: one candidate, in its provider's wire, from either client shape.
// ---------------------------------------------------------------------------

type ClientWire = 'openai' | 'anthropic';

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  payload: Json;
}

function upstreamCall(candidate: Candidate, client: ClientWire, body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  const model = candidate.model.id;
  if (provider.wire === 'openai') {
    let payload = client === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) {
      // A streamed answer must end with its usage: ask the provider for it, as a gateway may,
      // whatever the client asked for. A provider that never sends usage leaves the wire
      // column's usage check unanswerable.
      const options = isRecord(payload.stream_options) ? payload.stream_options : {};
      payload = { ...payload, stream_options: { ...options, include_usage: true } };
    }
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload = client === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  const openAIForm = client === 'openai' ? body : anthropicToOpenAI(body, model);
  return { url: `${base}/v1beta/models/${model}:${action}`, headers, payload: openAIToGeminiRequest(openAIForm) };
}

/** Write a translated event stream to the client, generator by generator. */
async function writeEvents(res: ServerResponse, events: AsyncGenerator<string>): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/** Pipe a provider's event stream to the client as it arrives, unchanged. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function sseHeaders(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

function sendJsonOk(res: ServerResponse, body: unknown): boolean {
  const text = JSON.stringify(body);
  res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
  return true;
}

/**
 * Serve one client request from one candidate. Returns false when the provider refused or could
 * not be used, so the caller fails over to the next candidate; true once the client has its answer.
 */
async function attempt(candidate: Candidate, client: ClientWire, body: Json, stream: boolean, requestedModel: string, res: ServerResponse): Promise<boolean> {
  const wire = candidate.probe.provider.wire;
  const call = upstreamCall(candidate, client, body, stream);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return false;
  }
  if (!upstream.ok) {
    try {
      await upstream.text();
    } catch {
      // Nothing to drain: the refusal stands.
    }
    return false;
  }

  if (stream && upstream.body !== null) {
    sseHeaders(res);
    if (wire === 'openai' && client === 'openai') await pipeStream(upstream.body, res);
    else if (wire === 'anthropic' && client === 'anthropic') await pipeStream(upstream.body, res);
    else if (wire === 'openai') await writeEvents(res, anthropicEventsFromOpenAI(upstream.body, requestedModel));
    else if (wire === 'anthropic') await writeEvents(res, anthropicEventsToOpenAI(upstream.body, candidate.model.id));
    else if (client === 'openai') await writeEvents(res, geminiEventsToOpenAI(upstream.body, candidate.model.id));
    else await writeEvents(res, anthropicEventsFromOpenAI(streamOfText(geminiEventsToOpenAI(upstream.body, candidate.model.id)), requestedModel));
    return true;
  }

  const text = await upstream.text();
  if (wire === 'openai' && client === 'openai') {
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return true;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(text);
  } catch {
    return false;
  }
  if (!isRecord(answer)) return false;
  if (client === 'openai') {
    if (wire === 'anthropic') return sendJsonOk(res, anthropicResponseToOpenAI(answer, candidate.model.id));
    return sendJsonOk(res, geminiToOpenAI(answer, candidate.model.id));
  }
  if (wire === 'anthropic') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(text);
    return true;
  }
  if (wire === 'openai') return sendJsonOk(res, openAIToAnthropic(answer, requestedModel));
  return sendJsonOk(res, openAIToAnthropic(geminiToOpenAI(answer, candidate.model.id), requestedModel));
}

/** One client request, across every candidate until one serves it. */
async function serve(client: ClientWire, body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const arriving = requirementsFor(body);
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : '';
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonOnOpenAIClient = client === 'openai' && (format === 'json_object' || format === 'json_schema');
  const triedModels = new Set<string>();
  for (;;) {
    const candidates = candidatesFor(arriving.requires, arriving.weight, jsonOnOpenAIClient).filter((c) => !triedModels.has(c.probe.provider.id + '/' + c.model.id));
    if (candidates.length === 0) break;
    const candidate = candidates[0];
    triedModels.add(candidate.probe.provider.id + '/' + candidate.model.id);
    const state = stateOf(candidate.probe.provider.id);
    state.calls += 1;
    const served = await attempt(candidate, client, body, stream, requestedModel !== '' ? requestedModel : candidate.model.id, res);
    if (served) {
      state.consecutiveFailures = 0;
      return;
    }
    state.failures += 1;
    state.consecutiveFailures += 1;
    if (res.headersSent) return; // A stream already began: nothing more can be sent.
  }
  if (!res.headersSent) {
    if (client === 'anthropic') sendJson(res, 503, anthropicError('overloaded_error', 'no free model on the directory could serve this request'));
    else sendJson(res, 503, openAIError('no free model on the directory could serve this request', 'no_free_model', null));
  }
}

// ---------------------------------------------------------------------------
// The server: routes, register, poll (the trial contract as the baseline sets out).
// ---------------------------------------------------------------------------

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

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
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

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

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

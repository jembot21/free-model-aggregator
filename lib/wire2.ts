// The upstream wires the baseline lacks: Anthropic and Gemini, in both directions, JSON and SSE.
//
// main.ts routes a client request (OpenAI Chat Completions or Anthropic Messages) to a provider on
// any of the three wires. wire.ts already converts between the two client shapes through the OpenAI
// form; this file adds the two missing upstream shapes, reusing that hub:
//
//   openai client  -> anthropic upstream: openAIToAnthropicRequest, anthropicToOpenAI (response),
//                                          anthropicEventsToOpenAI (stream)
//   openai client  -> gemini upstream:    openAIToGeminiRequest, geminiToOpenAI,
//                                          geminiEventsToOpenAI (stream)
//   anthropic client -> gemini upstream:  anthropicToOpenAI + openAIToGeminiRequest upstream, and
//                                          geminiToOpenAI + openAIToAnthropic back (streamed through
//                                          anthropicEventsFromOpenAI over a synthesized stream)
//
// Everything is a pure function of its arguments: fixed timestamps, synthesized ids derived from
// counters in the stream's own order, no clock and no randomness, so a replay reproduces the bytes.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A fixed epoch second for synthesized OpenAI responses: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

function num(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function jsonOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty object.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions -> Anthropic Messages (request)
// ---------------------------------------------------------------------------

/** One OpenAI image_url part as an Anthropic image block; null when it carries nothing usable. */
function imageBlock(part: Json): Json | null {
  const image = isRecord(part.image_url) ? part.image_url : null;
  if (image === null || typeof image.url !== 'string') return null;
  const url = image.url;
  const data = /^data:([^;,]+);base64,(.*)$/.exec(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: system
 * messages join into the system prompt, image_url parts become image blocks, assistant tool_calls
 * become tool_use blocks, tool messages become tool_result blocks inside a user message, and the
 * tools, tool choice, output cap, temperature, top_p and stop sequences are rewritten.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];

  const pushUserBlocks = (blocks: Json[]): void => {
    if (blocks.length === 0) return;
    const last = messages[messages.length - 1];
    if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) {
      (last.content as Json[]).push(...blocks);
      return;
    }
    messages.push({ role: 'user', content: blocks });
  };

  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system' || role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool' || role === 'function') {
      pushUserBlocks([
        {
          type: 'tool_result',
          tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
        },
      ]);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: jsonOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // user (and anything else) becomes a user message of text and image blocks.
    if (typeof raw.content === 'string') {
      pushUserBlocks([{ type: 'text', text: raw.content }]);
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url') {
          const block = imageBlock(part);
          if (block !== null) blocks.push(block);
        }
      }
      pushUserBlocks(blocks);
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  // The output cap passes through as the client set it; inventing one would misreport it.
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Anthropic Messages -> OpenAI Chat Completions (response)
// ---------------------------------------------------------------------------

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const prompt = num(usage.input_tokens);
  const completion = num(usage.output_tokens);
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReasonOf(body.stop_reason),
      },
    ],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/** One OpenAI streaming chunk in SSE form. */
function openAIChunk(delta: Json, finish: string | null, model: string, usage?: Json): string {
  const chunk: Json = {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
  if (usage !== undefined) chunk.usage = usage;
  return sseEvent(null, chunk);
}

/**
 * An Anthropic SSE stream as OpenAI Chat Completions chunks: a role chunk first, text and tool
 * argument deltas as they come, then a finish chunk carrying the stop reason, a usage chunk, and
 * [DONE] — the order a real OpenAI stream with include_usage ends in.
 */
export async function* anthropicEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolIndexByBlock = new Map<number, number>();
  let nextToolIndex = 0;

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk({ role: 'assistant', content: '' }, null, model)];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk({}, finishReasonOf(stopReason), model),
      sseEvent(null, {
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    let eventName = '';
    let dataText = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      else if (line.startsWith('data:')) dataText += line.slice(5).trim();
    }
    if (dataText === '') return out;
    let data: unknown;
    try {
      data = JSON.parse(dataText);
    } catch {
      return out;
    }
    if (!isRecord(data)) return out;
    out.push(...start());
    if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = num(data.message.usage.input_tokens);
    } else if (eventName === 'content_block_start' && isRecord(data.content_block)) {
      const blockIndex = num(data.index);
      if (data.content_block.type === 'tool_use') {
        const toolIndex = nextToolIndex;
        nextToolIndex += 1;
        toolIndexByBlock.set(blockIndex, toolIndex);
        out.push(
          openAIChunk(
            {
              tool_calls: [
                {
                  index: toolIndex,
                  id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${toolIndex}`,
                  type: 'function',
                  function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' },
                },
              ],
            },
            null,
            model,
          ),
        );
      }
    } else if (eventName === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        out.push(openAIChunk({ content: data.delta.text }, null, model));
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const toolIndex = toolIndexByBlock.get(num(data.index)) ?? 0;
        out.push(openAIChunk({ tool_calls: [{ index: toolIndex, function: { arguments: data.delta.partial_json } }] }, null, model));
      }
    } else if (eventName === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = num(data.usage.output_tokens);
    } else if (eventName === 'message_stop') {
      out.push(...finish());
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of start()) yield event;
    for (const event of finish()) yield event;
  }
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions -> Gemini generateContent (request)
// ---------------------------------------------------------------------------

/** One OpenAI content part as a Gemini part: text, an inline image, or a file image; null otherwise. */
function geminiPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+);base64,(.*)$/.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: system messages become the
 * system instruction, user and model turns become contents, tool answers become functionResponse
 * parts on a user turn, and tools, tool choice and generation settings are rewritten.
 */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];

  // Tool answers name their call by id; the functionResponse needs the call's name, so map first.
  const nameByCallId = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw) || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') {
        nameByCallId.set(call.id, call.function.name);
      }
    }
  }

  const pushContent = (role: 'user' | 'model', parts: Json[]): void => {
    if (parts.length === 0) return;
    const last = contents[contents.length - 1];
    if (last !== undefined && last.role === role) {
      (last.parts as Json[]).push(...parts);
      return;
    }
    contents.push({ role, parts });
  };

  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    if (raw.role === 'tool' || raw.role === 'function') {
      const name = (typeof raw.tool_call_id === 'string' && nameByCallId.get(raw.tool_call_id)) || (typeof raw.name === 'string' ? raw.name : 'tool');
      const content = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      let response: Json = { result: content };
      try {
        const parsed: unknown = JSON.parse(content);
        if (isRecord(parsed)) response = parsed;
      } catch {
        // Plain text result: wrap it.
      }
      pushContent('user', [{ functionResponse: { name, response } }]);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: jsonOf(call.function.arguments),
          },
        });
      }
      pushContent('model', parts);
      continue;
    }
    if (typeof raw.content === 'string') {
      pushContent('user', [{ text: raw.content }]);
      continue;
    }
    if (Array.isArray(raw.content)) {
      const parts: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const converted = geminiPart(part);
        if (converted !== null) parts.push(converted);
      }
      pushContent('user', parts);
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  } else if (choice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };

  const generation: Json = {};
  if (typeof body.max_tokens === 'number') generation.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') generation.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') generation.temperature = body.temperature;
  if (typeof body.top_p === 'number') generation.topP = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') generation.stopSequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) generation.stopSequences = stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') generation.responseMimeType = 'application/json';
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  return out;
}

// ---------------------------------------------------------------------------
// Gemini generateContent -> OpenAI Chat Completions (response and stream)
// ---------------------------------------------------------------------------

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinish(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'content_filter';
  return 'stop';
}

interface GeminiAnswer {
  text: string;
  toolCalls: Json[];
  finish: string | null;
  promptTokens: number;
  completionTokens: number;
}

/** The parts of one Gemini answer (a full response or one streamed chunk) in OpenAI terms. */
function geminiAnswer(body: Json): GeminiAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    finish: typeof candidate.finishReason === 'string' ? geminiFinish(candidate.finishReason) : null,
    promptTokens: num(usage.promptTokenCount),
    completionTokens: num(usage.candidatesTokenCount),
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const answer = geminiAnswer(body);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: answer.text !== '' ? answer.text : null,
          ...(answer.toolCalls.length > 0 ? { tool_calls: answer.toolCalls } : {}),
        },
        finish_reason: answer.finish ?? 'stop',
      },
    ],
    usage: {
      prompt_tokens: answer.promptTokens,
      completion_tokens: answer.completionTokens,
      total_tokens: answer.promptTokens + answer.completionTokens,
    },
  };
}

/**
 * A Gemini SSE stream as OpenAI Chat Completions chunks: a role chunk, the text and tool-call
 * deltas of each chunk, then the finish chunk, a usage chunk and [DONE].
 */
export async function* geminiEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let promptTokens = 0;
  let completionTokens = 0;
  let toolCount = 0;

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk({ role: 'assistant', content: '' }, null, model)];
  };
  const finish = (reason: string | null): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk({}, reason ?? 'stop', model),
      sseEvent(null, {
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const answer = geminiAnswer(chunk);
      promptTokens = answer.promptTokens || promptTokens;
      completionTokens = answer.completionTokens || completionTokens;
      out.push(...start());
      if (answer.text !== '') out.push(openAIChunk({ content: answer.text }, null, model));
      for (const call of answer.toolCalls) {
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          openAIChunk(
            {
              tool_calls: [
                {
                  index: toolCount,
                  id: `call_${toolCount}`,
                  type: 'function',
                  function: { name: typeof fn.name === 'string' ? fn.name : '', arguments: typeof fn.arguments === 'string' ? fn.arguments : '' },
                },
              ],
            },
            null,
            model,
          ),
        );
        toolCount += 1;
      }
      if (answer.finish !== null) out.push(...finish(answer.finish));
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of start()) yield event;
    for (const event of finish(null)) yield event;
  }
}

/** A ReadableStream that serves an async generator of strings: how a converted stream feeds another. */
export function streamOfText(source: AsyncGenerator<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await source.next();
      if (next.done === true) controller.close();
      else controller.enqueue(encoder.encode(next.value));
    },
    async cancel() {
      await source.return(undefined);
    },
  });
}

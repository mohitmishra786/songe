"""The Responses API (/v1/responses) on top of Chat Completions.

Codex CLI speaks only the Responses API. Songe serves it by translating: the
request becomes a chat completion, which goes through the same path as any
other (the learner's version, its lessons, the recording), and the reply is
translated back, streamed or not. So every model server Songe supports works,
whether or not it serves /v1/responses itself, and what Songe learns from
has one shape.

Only what an agent sends with `store: false` is supported: the whole
conversation in `input`. Hosted tools (web search, file search) are left out;
there is nothing local to run them.
"""

import json
import time
import uuid

# Tools are flattened to chat function names; a namespaced one (Codex's
# multi_agent_v1) gets its namespace in front, and the call is mapped back.
SEPARATOR = "__"


class Unsupported(ValueError):
    """A request this translation cannot serve faithfully."""


def text_of(content):
    """The text of a message's content: a string, or a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text") or ""
        for part in content or ()
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")
    )


def chat_content(content):
    """Content for a chat message: plain text, or parts when there is an image."""
    if isinstance(content, str) or not any(
        isinstance(p, dict) and p.get("type") == "input_image" for p in content or ()
    ):
        return text_of(content)
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "input_image" and part.get("image_url"):
            parts.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
        elif part.get("type") in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text") or ""})
    return parts


def output_text(output):
    """A tool call's output: a string, or a list of content items."""
    return output if isinstance(output, str) else text_of(output)


def tools_for_chat(tools):
    """(chat tools, {chat name: (kind, namespace, name)})."""
    chat, names = [], {}

    def add(tool, namespace=None):
        kind = tool.get("type")
        name = tool.get("name")
        if kind not in ("function", "custom") or not isinstance(name, str):
            return  # hosted tools: nothing local runs them
        flat = f"{namespace}{SEPARATOR}{name}" if namespace else name
        if kind == "function":
            parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        else:
            # A free-form tool (Codex's apply_patch on some models) takes one
            # string; as a function, that string is its one argument.
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            }
        function = {"name": flat, "parameters": parameters}
        if tool.get("description"):
            function["description"] = tool["description"]
        chat.append({"type": "function", "function": function})
        names[flat] = (kind, namespace, name)

    for tool in tools or ():
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            for inner in tool.get("tools") or ():
                if isinstance(inner, dict):
                    add(inner, tool.get("name"))
        else:
            add(tool)
    return chat, names


def flat_name(item):
    namespace = item.get("namespace")
    return f"{namespace}{SEPARATOR}{item['name']}" if namespace else item["name"]


def to_chat(body):
    """(chat completion request, tool names) for a Responses request."""
    if not isinstance(body, dict):
        raise Unsupported("the request is not a JSON object")
    if body.get("previous_response_id"):
        # Songe keeps no responses: the agent must send the whole conversation.
        raise Unsupported("previous_response_id needs stored responses; send store: false")
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": text_of(body["instructions"])})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items or ():
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            # Chat templates know system, not developer.
            role = "system" if role == "developer" else role
            content = item.get("content")
            messages.append(
                {"role": role, "content": text_of(content) if role == "assistant" else chat_content(content)}
            )
        elif kind in ("function_call", "custom_tool_call"):
            arguments = (
                item.get("arguments") or "{}"
                if kind == "function_call"
                else json.dumps({"input": item.get("input") or ""}, ensure_ascii=False)
            )
            call = {
                "id": item.get("call_id") or item.get("id") or "",
                "type": "function",
                "function": {"name": flat_name(item), "arguments": arguments},
            }
            previous = messages[-1] if messages else None
            # A reply's text and its calls, or several calls, are one turn.
            if previous and previous["role"] == "assistant" and "tool_calls" not in previous:
                previous["tool_calls"] = [call]
            elif previous and previous["role"] == "assistant":
                previous["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": output_text(item.get("output")),
                }
            )
        # Reasoning items (encrypted, for OpenAI's own models) and hosted
        # tool calls have nothing a local model can read.
    chat = {"model": body.get("model"), "messages": messages}
    tools, names = tools_for_chat(body.get("tools"))
    if tools:
        chat["tools"] = tools
        choice = body.get("tool_choice")
        if isinstance(choice, str):
            chat["tool_choice"] = choice
        elif isinstance(choice, dict) and choice.get("name"):
            chat["tool_choice"] = {"type": "function", "function": {"name": flat_name(choice)}}
        if "parallel_tool_calls" in body:
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    for ours, theirs in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(ours) is not None:
            chat[theirs] = body[ours]
    form = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
    if isinstance(form, dict) and form.get("type") == "json_schema":
        chat["response_format"] = {
            "type": "json_schema",
            "json_schema": {k: form[k] for k in ("name", "schema", "strict") if k in form},
        }
    if body.get("stream"):
        chat["stream"] = True
        chat["stream_options"] = {"include_usage": True}
    return chat, names


def call_item(call, names, index):
    """A chat tool call as a Responses output item."""
    function = call.get("function") or {}
    flat = function.get("name") or ""
    kind, namespace, name = names.get(flat, ("function", None, flat))
    call_id = call.get("id") or f"call_{index}"
    if kind == "custom":
        try:
            value = json.loads(function.get("arguments") or "{}").get("input", "")
        except (ValueError, AttributeError):
            value = function.get("arguments") or ""
        return {"type": "custom_tool_call", "id": f"ctc_{call_id}", "status": "completed",
                "call_id": call_id, "name": name, "input": value}
    item = {"type": "function_call", "id": f"fc_{call_id}", "status": "completed",
            "call_id": call_id, "name": name, "arguments": function.get("arguments") or ""}
    if namespace:
        item["namespace"] = namespace
    return item


def usage(chat_usage):
    if not isinstance(chat_usage, dict):
        return None
    prompt = chat_usage.get("prompt_tokens") or 0
    completion = chat_usage.get("completion_tokens") or 0
    cached = (chat_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (chat_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": chat_usage.get("total_tokens") or prompt + completion,
    }


def response(model, output, status="completed", finish=None, used=None, ident=None):
    result = {
        "id": ident or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "usage": used,
        "error": None,
        "incomplete_details": None,
    }
    if finish == "length" and status == "completed":
        result["status"] = "incomplete"
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    return result


def reasoning_item(text, ident):
    return {"type": "reasoning", "id": ident, "summary": [],
            "content": [{"type": "reasoning_text", "text": text}]}


def message_item(text, ident, status="completed"):
    return {"type": "message", "id": ident, "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def from_chat(chat, model, names):
    """A Responses object for a (non-streamed) chat completion."""
    choice = ((chat or {}).get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    thought = message.get("reasoning_content") or message.get("reasoning")
    if thought:
        output.append(reasoning_item(thought, "rs_0"))
    if message.get("content"):
        output.append(message_item(text_of(message["content"]), "msg_0"))
    for index, call in enumerate(message.get("tool_calls") or ()):
        output.append(call_item(call, names, index))
    return response(model, output, finish=choice.get("finish_reason"), used=usage(chat.get("usage")))


def event(kind, **fields):
    data = json.dumps({"type": kind, **fields}, ensure_ascii=False)
    return f"event: {kind}\ndata: {data}\n\n".encode()


class StreamTranslator:
    """Chat completion SSE bytes in, Responses SSE events out."""

    def __init__(self, model, names):
        self.model, self.names = model, names
        self.ident = f"resp_{uuid.uuid4().hex}"
        self.buffer = ""
        self.sequence = 0
        self.output = []  # items, in output order
        self.text = None  # (index, item) of the open message
        self.thought = None  # (index, item) of the open reasoning
        self.calls = {}  # chat tool call index -> output index
        self.finish = None
        self.used = None
        self.done = False

    def emit(self, kind, **fields):
        self.sequence += 1
        return event(kind, sequence_number=self.sequence, **fields)

    def start(self):
        shell = response(self.model, [], status="in_progress", ident=self.ident)
        return self.emit("response.created", response=shell) + self.emit(
            "response.in_progress", response=shell
        )

    def feed(self, chunk):
        self.buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
        out = []
        while "\n\n" in self.buffer:
            raw, self.buffer = self.buffer.split("\n\n", 1)
            data = "\n".join(
                line[5:].lstrip(" ") for line in raw.splitlines() if line.startswith("data:")
            )
            if not data:
                continue
            if data == "[DONE]":
                self.done = True
                continue
            try:
                out.append(self.chunk(json.loads(data)))
            except ValueError:
                continue
        return b"".join(out)

    def open(self, item):
        self.output.append(item)
        index = len(self.output) - 1
        return index, self.emit("response.output_item.added", output_index=index, item=item)

    def chunk(self, data):
        out = []
        if data.get("usage"):
            self.used = usage(data["usage"])
        for choice in data.get("choices") or ():
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                if self.thought is None:
                    item = {"type": "reasoning", "id": "rs_0", "summary": [], "content": []}
                    index, added = self.open(item)
                    self.thought = (index, item)
                    item["content"].append({"type": "reasoning_text", "text": ""})
                    out.append(added)
                index, item = self.thought
                item["content"][0]["text"] += thought
                out.append(self.emit("response.reasoning_text.delta", item_id=item["id"],
                                     output_index=index, content_index=0, delta=thought))
            if delta.get("content"):
                out.append(self.close_thought())
                if self.text is None:
                    item = {"type": "message", "id": "msg_0", "role": "assistant",
                            "status": "in_progress", "content": []}
                    index, added = self.open(item)
                    self.text = (index, item)
                    part = {"type": "output_text", "text": "", "annotations": []}
                    item["content"].append(part)
                    out.append(added + self.emit("response.content_part.added", item_id="msg_0",
                                                 output_index=index, content_index=0,
                                                 part={**part}))
                index, item = self.text
                item["content"][0]["text"] += delta["content"]
                out.append(self.emit("response.output_text.delta", item_id="msg_0",
                                     output_index=index, content_index=0, delta=delta["content"]))
            for call in delta.get("tool_calls") or ():
                out.append(self.close_thought())
                position = call.get("index", 0)
                function = call.get("function") or {}
                if position not in self.calls:
                    # The chat names, kept to build the final item; the event
                    # shows the call as the agent named it.
                    item = {"type": "call", "call_id": call.get("id") or f"call_{position}",
                            "name": function.get("name") or "", "arguments": ""}
                    shown = call_item({"id": item["call_id"], "function": {"name": item["name"]}},
                                      self.names, position)
                    item["id"], item["kind"] = shown["id"], shown["type"]
                    shown.update(status="in_progress", **({"arguments": ""} if "arguments" in shown else {"input": ""}))
                    self.output.append(item)
                    self.calls[position] = len(self.output) - 1
                    out.append(self.emit("response.output_item.added",
                                         output_index=self.calls[position], item=shown))
                item = self.output[self.calls[position]]
                if function.get("name") and not item["name"]:
                    item["name"] = function["name"]
                if function.get("arguments"):
                    item["arguments"] += function["arguments"]
                    if item["kind"] == "function_call":
                        out.append(self.emit("response.function_call_arguments.delta", item_id=item["id"],
                                             output_index=self.calls[position], delta=function["arguments"]))
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return b"".join(out)

    def close_thought(self):
        if self.thought is None or self.thought[1].get("closed"):
            return b""
        index, item = self.thought
        item["closed"] = True
        text = item["content"][0]["text"]
        return self.emit("response.reasoning_text.done", item_id=item["id"], output_index=index,
                         content_index=0, text=text) + self.emit(
            "response.output_item.done", output_index=index, item=reasoning_item(text, item["id"]))

    def end(self):
        """The closing events; response.failed when the stream was cut short."""
        out = [self.close_thought()]
        final = []
        for index, item in enumerate(self.output):
            if item["type"] == "reasoning":
                final.append(reasoning_item(item["content"][0]["text"], item["id"]))
            elif item["type"] == "message":
                text = item["content"][0]["text"]
                done = message_item(text, item["id"])
                out.append(
                    self.emit("response.output_text.done", item_id=item["id"], output_index=index,
                              content_index=0, text=text)
                    + self.emit("response.content_part.done", item_id=item["id"], output_index=index,
                                content_index=0, part=done["content"][0])
                    + self.emit("response.output_item.done", output_index=index, item=done)
                )
                final.append(done)
            else:
                done = call_item(
                    {"id": item["call_id"], "function": {"name": item["name"], "arguments": item["arguments"]}},
                    self.names, index,
                )
                if done["type"] == "function_call":
                    out.append(self.emit("response.function_call_arguments.done", item_id=item["id"],
                                         output_index=index, arguments=item["arguments"]))
                out.append(self.emit("response.output_item.done", output_index=index, item=done))
                final.append(done)
        if not (self.done or self.finish):
            failed = response(self.model, final, status="failed", ident=self.ident)
            failed["error"] = {"code": "server_error", "message": "the model server's stream ended early"}
            out.append(self.emit("response.failed", response=failed))
        else:
            out.append(self.emit("response.completed", response=response(
                self.model, final, finish=self.finish, used=self.used, ident=self.ident)))
        return b"".join(out)

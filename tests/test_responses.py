import json

import httpx
import pytest

from songe.proxy import create_app
from songe.responses import StreamTranslator, Unsupported, to_chat
from tests.test_proxy import Chunks, pair
from tests.test_store import version

# The shape of a Codex CLI 0.172 request (captured), cut down.
CODEX = {
    "model": "songe/test",
    "stream": True,
    "input": [
        {"type": "message", "role": "developer",
         "content": [{"type": "input_text", "text": "You are a coding agent."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say hi"}]},
        {"type": "function_call", "id": "fc_1", "name": "exec_command",
         "arguments": '{"cmd": "echo hi"}', "call_id": "call_1"},
        {"type": "function_call_output", "call_id": "call_1", "output": "hi\n"},
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
    ],
    "tools": [
        {"type": "function", "name": "exec_command", "description": "Runs a command.",
         "strict": False, "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
        {"type": "namespace", "name": "multi_agent_v1", "description": "Sub-agents.",
         "tools": [{"type": "function", "name": "close_agent", "parameters": {"type": "object"}}]},
        {"type": "web_search", "external_web_access": False},
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "reasoning": {"summary": "auto"},
    "store": False,
    "include": ["reasoning.encrypted_content"],
}


def test_a_codex_request_becomes_a_chat_completion():
    chat, names = to_chat(CODEX)
    assert chat["messages"] == [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "exec_command", "arguments": '{"cmd": "echo hi"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "hi\n"},
    ]
    assert [t["function"]["name"] for t in chat["tools"]] == [
        "exec_command", "multi_agent_v1__close_agent"  # web_search: nothing local runs it
    ]
    assert names["multi_agent_v1__close_agent"] == ("function", "multi_agent_v1", "close_agent")
    assert chat["tool_choice"] == "auto" and chat["parallel_tool_calls"] is True
    assert chat["stream"] and chat["stream_options"] == {"include_usage": True}
    assert "reasoning" not in chat and "store" not in chat


def test_text_and_calls_of_one_reply_are_one_assistant_turn():
    chat, _ = to_chat({"model": "m", "instructions": "Be brief.", "max_output_tokens": 64, "input": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
        {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
        {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "c2"},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "Done"}]},
    ]})
    assert chat["messages"][0] == {"role": "system", "content": "Be brief."}
    turn = chat["messages"][1]
    assert turn["content"] == "Looking."
    assert [c["function"] for c in turn["tool_calls"]] == [
        {"name": "a", "arguments": "{}"},
        {"name": "apply_patch", "arguments": '{"input": "*** Begin Patch"}'},
    ]
    assert chat["messages"][2] == {"role": "tool", "tool_call_id": "c2", "content": "Done"}
    assert chat["max_tokens"] == 64


def test_stored_responses_are_refused():
    with pytest.raises(Unsupported, match="store: false"):
        to_chat({"model": "m", "input": "hi", "previous_response_id": "resp_1"})


def chat_sse(*chunks, done=True):
    wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    return wire + (b"data: [DONE]\n\n" if done else b"")


def delta(finish=None, **fields):
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": finish}]}


def events(body):
    found = []
    for block in body.decode().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines:
            found.append(json.loads(lines["data"]))
    return found


async def post(app, payload, headers=None):
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        return await client.post("/v1/responses", json=payload, headers=headers or {})


async def test_codex_streams_through_songe_and_the_chat_is_recorded(config, store):
    wire = chat_sse(
        delta(role="assistant", reasoning_content="Think."),
        delta(content="Hel"),
        delta(content="lo"),
        delta(tool_calls=[{"index": 0, "id": "call_9", "type": "function",
                           "function": {"name": "multi_agent_v1__close_agent", "arguments": '{"id":'}}]),
        delta("tool_calls", tool_calls=[{"index": 0, "function": {"arguments": '"a"}'}}]),
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    seen = []

    async def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Chunks([wire[i:i + 9] for i in range(0, len(wire), 9)]),
                              headers={"content-type": "text/event-stream"})

    store.add_version(version("night-1", 1))
    store.swap("night-1", None)
    app, upstream = await pair(config, store, handler)
    response = await post(app, CODEX, {"session-id": "s1", "accept-encoding": "zstd,gzip"})
    await upstream.aclose()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # The learner's version serves it, through the chat path.
    assert seen[0].url.path == "/v1/chat/completions"
    sent = json.loads(seen[0].content)
    assert sent["model"] == "night-1" and sent["messages"][1]["content"] == "say hi"
    assert seen[0].headers["accept-encoding"] == "identity"
    assert seen[0].headers["x-session-affinity"] == "s1"
    found = events(response.content)
    kinds = [e["type"] for e in found]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert kinds[-1] == "response.completed"
    assert "".join(e["delta"] for e in found if e["type"] == "response.output_text.delta") == "Hello"
    assert [e["sequence_number"] for e in found] == list(range(1, len(found) + 1))
    final = found[-1]["response"]
    assert final["model"] == "songe/test"
    assert final["output"][0]["content"] == [{"type": "reasoning_text", "text": "Think."}]
    assert final["output"][1]["content"][0]["text"] == "Hello"
    assert final["output"][2] == {
        "type": "function_call", "id": "fc_call_9", "status": "completed", "call_id": "call_9",
        "name": "close_agent", "arguments": '{"id":"a"}', "namespace": "multi_agent_v1",
    }
    added = next(e for e in found if e["type"] == "response.output_item.added"
                 and e["item"]["type"] == "function_call")
    assert added["item"]["name"] == "close_agent"
    assert final["usage"]["input_tokens"] == 7 and final["usage"]["output_tokens"] == 3
    # Recorded as the chat completion it was, in the shape Songe learns from.
    logged = store.exchanges()[0]
    assert logged["request"]["messages"] == to_chat(CODEX)[0]["messages"]
    assert logged["response"]["choices"][0]["message"]["content"] == "Hello"


async def test_a_reply_that_is_not_streamed(config, store):
    answer = {
        "choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "apply_patch", "arguments": '{"input": "patch"}'}}]}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    async def handler(request):
        return httpx.Response(200, stream=Chunks([json.dumps(answer).encode()]),
                              headers={"content-type": "application/json"})

    app, upstream = await pair(config, store, handler)
    payload = {"model": "songe/test", "input": "hi",
               "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar"}}]}
    response = await post(app, payload)
    await upstream.aclose()
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["output"] == [{"type": "custom_tool_call", "id": "ctc_c1", "status": "completed",
                               "call_id": "c1", "name": "apply_patch", "input": "patch"}]
    assert store.exchanges()[0]["request"]["messages"] == [{"role": "user", "content": "hi"}]


def test_a_stream_cut_short_fails_instead_of_completing():
    translator = StreamTranslator("songe/test", {})
    translator.start()
    translator.feed(chat_sse(delta(content="Hal"), done=False))
    kinds = [e["type"] for e in events(translator.end())]
    assert kinds[-1] == "response.failed"
    assert "response.completed" not in kinds


async def test_errors_pass_through_and_bad_requests_are_refused(config, store):
    async def handler(request):
        return httpx.Response(500, stream=Chunks([b'{"error": {"message": "boom"}}']),
                              headers={"content-type": "application/json"})

    app, upstream = await pair(config, store, handler)
    response = await post(app, {"model": "songe/test", "input": "hi"})
    assert response.status_code == 500 and response.json()["error"]["message"] == "boom"
    response = await post(app, {"model": "songe/test", "input": "hi", "previous_response_id": "r"})
    assert response.status_code == 400
    assert "store: false" in response.json()["error"]["message"]
    await upstream.aclose()


def test_the_route_exists(config, store):
    app = create_app(config, store)
    assert any(getattr(r, "path", None) == "/v1/responses" for r in app.routes)

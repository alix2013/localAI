"""
MCP client
- multi-server
- Bearer auth
- JSON Schema validation
- schema-version caching
- LLM integration
"""

import os
import json
import time
import requests
from jsonschema import validate, ValidationError
from typing import List, Dict, Any, Optional

# -------------------------
# Config
# -------------------------
# examples
MCP_SERVER_URLS = [
    "http://localhost:8000",  # basic (get_time, echo)
    "http://localhost:8001",  # math (add_numbers, multiply_numbers)
]

MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "super-secret-token")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-...")
LLM_MODEL = "gpt-4o-mini"

REQUEST_TIMEOUT = 6
RETRY_DELAY = 0.5
MAX_RETRIES = 2

# -------------------------
# HTTP helpers
# -------------------------
def http_get(url: str, timeout=REQUEST_TIMEOUT) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {MCP_BEARER_TOKEN}"}
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            raise

def http_post(url: str, json_payload: Dict[str, Any], timeout=REQUEST_TIMEOUT) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {MCP_BEARER_TOKEN}"}
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=json_payload, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            try:
                return r.json()
            except Exception:
                raise

# -------------------------
# Tool discovery + caching
# -------------------------
tool_cache: Dict[str, Dict[str, Any]] = {}  # server_url -> {"version":..., "tools": [...]}

def discover_tools(server_urls: List[str]) -> List[Dict[str, Any]]:
    discovered = []
    for url in server_urls:
        try:
            _ = http_get(f"{url}/initialize")
        except Exception as e:
            print(f"[WARN] Cannot initialize {url}: {e}")
            continue
        try:
            tools_resp = http_get(f"{url}/list_tools")
            tools = tools_resp.get("tools", [])
            version = tools_resp.get("tool_schema_version")
        except Exception as e:
            print(f"[WARN] Cannot list tools from {url}: {e}")
            continue

        cached = tool_cache.get(url)
        if cached and cached.get("version") == version:
            tools = cached["tools"]
            print(f"[INFO] Using cached tools for {url} (version {version})")
        else:
            tool_cache[url] = {"version": version, "tools": tools}
            print(f"[INFO] Cached tools for {url} (version {version})")

        for t in tools:
            t_copy = dict(t)
            t_copy["_server_url"] = url
            discovered.append(t_copy)
    return discovered

# -------------------------
# System prompt builder
# -------------------------
def build_system_prompt_from_tools(tools: List[Dict[str, Any]]) -> str:
    lines = [
        "You are an assistant that can call external tools (MCP servers).",
        "You MUST respond in one of two ways:",
        "1) If the user's request can be answered directly, reply in plain text.",
        "2) If you need a tool, reply ONLY with strict JSON in this exact shape:",
        '{"action":"use_tool","tool":"<tool_name>","server_url":"<server_url>","args":{...}}',
        "",
        "Available tools (name, server, description, args schema):",
    ]
    for t in tools:
        args_schema = json.dumps(t.get("args_schema", {}), ensure_ascii=False)
        result_schema = json.dumps(t.get("result_schema", {}), ensure_ascii=False)
        lines.append(f"- {t['name']}  (server: {t['_server_url']})")
        lines.append(f"  Description: {t.get('description','')}")
        lines.append(f"  Args schema: {args_schema}")
        lines.append(f"  Returns: {result_schema}")
        lines.append("")
    lines.append(
        "RULES:\n- Return EXACT JSON (no surrounding text) when using a tool.\n"
        "- Use the provided argument names and types from the schema.\n"
        "- If you cannot find a suitable tool, return: {\"action\":\"none\"}."
    )
    return "\n".join(lines)

# -------------------------
# LLM call
# -------------------------
def call_llm(system_prompt: str, user_text: str, temperature: float = 0.0) -> str:
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": 800,
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    parsed = resp.json()
    try:
        return parsed["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(parsed)

# -------------------------
# Call tool
# -------------------------
def call_tool_on_server(server_url: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"tool": tool_name, "args": args}
    return http_post(f"{server_url}/call_tool", payload)

def validate_args_against_tool(tool: Dict[str, Any], args: Dict[str, Any]) -> Optional[str]:
    schema = tool.get("args_schema", {"type":"object"})
    try:
        validate(instance=args, schema=schema)
        return None
    except ValidationError as ve:
        return ve.message

# -------------------------
# Agent loop
# -------------------------
def agent_loop(server_urls: List[str]):
    tools = discover_tools(server_urls)
    if not tools:
        print("No tools discovered. Exiting.")
        return
    system_prompt = build_system_prompt_from_tools(tools)
    print("System prompt built. Available tools:", [t["name"] for t in tools])
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit","exit"}:
            break

        try:
            llm_reply = call_llm(system_prompt, user_input)
        except Exception as e:
            print("[ERROR] LLM call failed:", e)
            continue

        try:
            decision = json.loads(llm_reply)
        except json.JSONDecodeError:
            print("Assistant:", llm_reply)
            continue

        action = decision.get("action")
        if action == "none":
            print("Assistant: Sorry, no suitable tool.")
            continue
        if action != "use_tool":
            print("Assistant returned unknown action:", decision)
            continue

        tool_name = decision.get("tool")
        server_url = decision.get("server_url")
        args = decision.get("args", {})

        if not tool_name or not server_url:
            print("[ERROR] Incomplete tool call info:", decision)
            continue

        tool_def = next((t for t in tools if t["name"]==tool_name and t["_server_url"]==server_url), None)
        if not tool_def:
            print("[ERROR] Tool not found on server.")
            continue

        err = validate_args_against_tool(tool_def, args)
        if err:
            print(f"[ERROR] Argument validation failed: {err}")
            continue

        try:
            tool_result = call_tool_on_server(server_url, tool_name, args)
        except Exception as e:
            print(f"[ERROR] Tool call failed: {e}")
            continue

        print("Tool result:", json.dumps(tool_result, ensure_ascii=False))

        # optional: feed result back to LLM
        feedback_prompt = f"User asked: {user_input}\nTool {tool_name} returned: {json.dumps(tool_result, ensure_ascii=False)}\nProvide a concise reply."
        try:
            final_reply = call_llm(system_prompt, feedback_prompt, temperature=0.4)
            print("Assistant:", final_reply)
        except Exception as e:
            print("[WARN] followup LLM call failed:", e)
            print("Assistant (raw tool result):", json.dumps(tool_result, ensure_ascii=False))

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    print("Discovering tools from MCP servers:", MCP_SERVER_URLS)
    agent_loop(MCP_SERVER_URLS)


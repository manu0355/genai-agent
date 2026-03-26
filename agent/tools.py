"""
Tool definitions: API, terminal, code execution, folder access, and memory.
Each tool exposes a callable and an Ollama-compatible schema dict.
"""

import subprocess
import sys
import textwrap
import traceback
import httpx

from .memory import PermanentMemory


# ---------------------------------------------------------------------------
# 2.1 API tool
# ---------------------------------------------------------------------------

def api_request(method: str, url: str, headers: dict | None = None, body: dict | None = None) -> str:
    """Make an HTTP request and return the response body as text."""
    try:
        resp = httpx.request(
            method=method.upper(),
            url=url,
            headers=headers or {},
            json=body if body else None,
            timeout=20,
            follow_redirects=True,
        )
        return f"Status: {resp.status_code}\n{resp.text[:4000]}"
    except Exception as exc:
        return f"Error: {exc}"


API_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "api_request",
        "description": "Make an HTTP request to any URL (GET, POST, PUT, DELETE, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "method":  {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE…)"},
                "url":     {"type": "string", "description": "Full URL to request"},
                "headers": {"type": "object", "description": "Optional HTTP headers as key-value pairs"},
                "body":    {"type": "object", "description": "Optional JSON body for POST/PUT requests"},
            },
            "required": ["method", "url"],
        },
    },
}


# ---------------------------------------------------------------------------
# 2.2 Terminal tool
# ---------------------------------------------------------------------------

def run_terminal(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if not output.strip():
            output = "(no output)"
        return output[:8000]
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


TERMINAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": "Execute a shell command and return its output. Use for file management, git, package installs, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
            },
            "required": ["command"],
        },
    },
}


# ---------------------------------------------------------------------------
# 2.3 Code execution tool
# ---------------------------------------------------------------------------

def execute_code(code: str, language: str = "python") -> str:
    """Execute Python code in a subprocess and return stdout + stderr."""
    if language.lower() != "python":
        return f"Only 'python' is supported, got {language!r}."
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout or ""
        err = result.stderr or ""
        if result.returncode != 0:
            return f"Exit code {result.returncode}\n{out}\n[stderr]\n{err}".strip()
        return (out or "(no output)") + (f"\n[stderr]\n{err}" if err else "")
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out after 30s"
    except Exception:
        return traceback.format_exc()


CODE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": "Execute Python code and return the output. Useful for calculations, data processing, and scripting.",
        "parameters": {
            "type": "object",
            "properties": {
                "code":     {"type": "string", "description": "Python code to execute"},
                "language": {"type": "string", "description": "Language (only 'python' is supported)"},
            },
            "required": ["code"],
        },
    },
}



# ---------------------------------------------------------------------------
# 3. Memory tools (wraps PermanentMemory instance)
# ---------------------------------------------------------------------------

def make_memory_tools(mem: PermanentMemory) -> tuple[dict, list[dict]]:
    """Return (tool_registry additions, schema list) bound to the given memory."""

    def remember(key: str, value: str) -> str:
        return mem.remember(key, value)

    def recall(key: str) -> str:
        return mem.recall(key)

    def forget(key: str) -> str:
        return mem.forget(key)

    def list_memories() -> str:
        return mem.list_memories()

    registry = {
        "remember": remember,
        "recall": recall,
        "forget": forget,
        "list_memories": list_memories,
    }

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Permanently store a fact that should persist across conversations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key":   {"type": "string", "description": "Short label for this fact"},
                        "value": {"type": "string", "description": "The fact to store"},
                    },
                    "required": ["key", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Retrieve a previously stored fact by its key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The key of the fact to retrieve"},
                    },
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": "Remove a permanently stored fact by its key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The key to forget"},
                    },
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_memories",
                "description": "List all permanently stored facts.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    return registry, schemas


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_tool_registry(mem: PermanentMemory) -> tuple[dict, list[dict]]:
    """Return a (name -> callable) registry and a list of Ollama tool schemas."""
    memory_registry, memory_schemas = make_memory_tools(mem)

    registry = {
        "api_request":   api_request,
        "run_terminal":  run_terminal,
        "execute_code":  execute_code,
        **memory_registry,
    }

    schemas = [
        API_TOOL_SCHEMA,
        TERMINAL_TOOL_SCHEMA,
        CODE_TOOL_SCHEMA,
        *memory_schemas,
    ]

    return registry, schemas

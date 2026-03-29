"""
Agent core — tool-calling loop over a pluggable LLM backend.
"""

from typing import Iterator

from rich.console import Console

from .backends import Backend, make_backend
from .memory import PermanentMemory, TemporaryMemory
from .tools import build_tool_registry

_console = Console(stderr=True)

DEFAULT_MODEL = "deepseek-coder:6.7b"
DEFAULT_BACKEND = "ollama"

SYSTEM_PROMPT = """You are a capable AI assistant with access to the following tools:

- api_request: Make HTTP requests to external APIs
- run_terminal: Execute shell commands
- execute_code: Run Python code
- search: Search the web (type='web', DuckDuckGo) or query a PostgreSQL database (type='db', SQL)
- remember / recall / forget / list_memories: Manage your persistent memory across sessions

{memory_context}

Guidelines:
- Use tools whenever they help you give a better answer.
- Prefer precise tool calls over guessing.
- When asked to remember something, always call the `remember` tool.
- When the user mentions something you should know in the future, proactively store it.
- Be concise in your final answers.
"""


class Agent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        backend: str = DEFAULT_BACKEND,
        memory_path: str = "memory_store.json",
    ):
        self._backend: Backend = make_backend(backend, model)
        self.model = model
        self.backend_name = backend
        self.permanent = PermanentMemory(memory_path)
        self.temporary = TemporaryMemory()
        self._registry, self._schemas = build_tool_registry(self.permanent)

    def _system_message(self) -> str:
        ctx = self.permanent.as_context_string()
        return SYSTEM_PROMPT.format(memory_context=ctx)

    def _build_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self._system_message()},
            *self.temporary.get_messages(),
        ]

    def _call_tool(self, name: str, args: dict) -> str:
        fn = self._registry.get(name)
        if fn is None:
            _console.print(f"[bold red]✗ unknown tool:[/bold red] {name!r}")
            return f"Unknown tool: {name!r}"
        _console.print(f"[bold yellow]⚙ tool:[/bold yellow] [cyan]{name}[/cyan]  args={args}")
        try:
            result = str(fn(**args))
            _console.print(f"[dim]  → {result[:200]}{'…' if len(result) > 200 else ''}[/dim]")
            return result
        except Exception as exc:
            _console.print(f"[bold red]  → error:[/bold red] {exc}")
            return f"Tool error: {exc}"

    def _run_tool_loop(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """
        Execute tool calls until the model stops requesting them.
        Returns (updated messages, tool_calls_were_made).
        """
        tool_calls_made = False
        while True:
            response = self._backend.chat(messages, self._schemas)

            if not response.tool_calls:
                if not tool_calls_made:
                    # First response, no tools — caller should use response.content directly.
                    messages.append({"role": "assistant", "content": response.content})
                return messages, tool_calls_made

            tool_calls_made = True
            messages.append({"role": "assistant", "content": response.content})

            for tc in response.tool_calls:
                result = self._call_tool(tc.name, tc.arguments)
                messages.append({"role": "tool", "content": result})

    def chat(self, user_input: str) -> str:
        """Non-streaming single turn."""
        self.temporary.add("user", user_input)
        messages = self._build_messages()
        messages, _ = self._run_tool_loop(messages)
        final = messages[-1]["content"]
        self.temporary.add("assistant", final)
        return final

    def stream_chat(self, user_input: str) -> Iterator[str]:
        """Stream response tokens. Tool calls run silently before streaming begins."""
        self.temporary.add("user", user_input)
        messages = self._build_messages()
        messages, tool_calls_made = self._run_tool_loop(messages)

        if not tool_calls_made:
            # The tool loop already has the final answer appended.
            content = messages[-1]["content"]
            self.temporary.add("assistant", content)
            yield content
            return

        # Tool calls were made — stream a fresh synthesis.
        full = ""
        for token in self._backend.stream(messages, self._schemas):
            full += token
            yield token
        self.temporary.add("assistant", full)

    def reset_session(self) -> None:
        """Clear conversation history (permanent memory is untouched)."""
        self.temporary.clear()

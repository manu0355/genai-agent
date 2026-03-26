"""
Interactive CLI for the agent.

Usage:
  uv run python main.py                          # Ollama, default model
  uv run python main.py --model qwen3.5:0.8b     # Ollama, specific model
  uv run python main.py --backend xai --model grok-3-mini   # xAI Grok
"""

import argparse

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
from rich.prompt import Prompt

from agent import Agent
from agent.core import DEFAULT_BACKEND, DEFAULT_MODEL

console = Console()


def print_banner(model: str, backend: str) -> None:
    console.print(f"\n[bold cyan]Agent[/bold cyan] — backend: [magenta]{backend}[/magenta]  model: [yellow]{model}[/yellow]")
    console.print("Commands: [bold]/reset[/bold] clear session  [bold]/memory[/bold] list facts  [bold]/quit[/bold] exit\n")


def run(model: str, backend: str, memory_path: str = "memory_store.json") -> None:
    try:
        agent = Agent(model=model, backend=backend, memory_path=memory_path)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return

    print_banner(model, backend)

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input.strip():
            continue

        if user_input.strip() == "/quit":
            console.print("[dim]Bye![/dim]")
            break
        elif user_input.strip() == "/reset":
            agent.reset_session()
            console.print("[dim]Session cleared.[/dim]")
            continue
        elif user_input.strip() == "/memory":
            console.print(agent.permanent.list_memories())
            continue

        console.print("\n[bold blue]Agent[/bold blue]", end=" ")
        try:
            for token in agent.stream_chat(user_input):
                console.print(token, end="", markup=False)
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
        console.print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local/cloud LLM agent")
    parser.add_argument("--model",   default=DEFAULT_MODEL,   help="Model name")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["ollama", "xai"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--memory",  default="memory_store.json", help="Memory file path")
    args = parser.parse_args()
    run(model=args.model, backend=args.backend, memory_path=args.memory)

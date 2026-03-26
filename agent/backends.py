"""
LLM backend abstraction.

Supported backends:
  - OllamaBackend  — local models via Ollama
  - XAIBackend     — xAI Grok via OpenAI-compatible API (requires XAI_API_KEY)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import ollama
from openai import OpenAI


# ---------------------------------------------------------------------------
# Unified data types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Backend(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        """Non-streaming call; returns a unified ChatResponse."""

    @abstractmethod
    def stream(self, messages: list[dict], tools: list[dict]) -> Iterator[str]:
        """Streaming call; yields text tokens."""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(Backend):
    def __init__(self, model: str):
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        resp = ollama.chat(model=self.model, messages=messages, tools=tools)
        msg = resp.message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            args = tc.function.arguments or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args))
        return ChatResponse(content=msg.content or "", tool_calls=tool_calls)

    def stream(self, messages: list[dict], tools: list[dict]) -> Iterator[str]:
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            yield chunk.message.content or ""


# ---------------------------------------------------------------------------
# xAI (Grok) backend  — OpenAI-compatible
# ---------------------------------------------------------------------------

XAI_BASE_URL = "https://api.x.ai/v1"
XAI_DEFAULT_MODEL = "grok-3-mini"


class XAIBackend(Backend):
    def __init__(self, model: str = XAI_DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        key = api_key or os.environ.get("XAI_API_KEY")
        if not key:
            raise ValueError("XAI_API_KEY environment variable is not set.")
        self._client = OpenAI(api_key=key, base_url=XAI_BASE_URL)

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools or None,  # type: ignore[arg-type]
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            args = tc.function.arguments or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args))
        return ChatResponse(content=msg.content or "", tool_calls=tool_calls)

    def stream(self, messages: list[dict], tools: list[dict]) -> Iterator[str]:
        for chunk in self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
        ):
            yield chunk.choices[0].delta.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(backend: str, model: str) -> Backend:
    """
    backend: 'ollama' | 'xai'
    model:   model name/id appropriate for the backend
    """
    if backend == "xai":
        return XAIBackend(model=model)
    return OllamaBackend(model=model)

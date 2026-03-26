"""
Permanent (file-based JSON) and temporary (in-session) memory.
"""

import json
from pathlib import Path
from datetime import datetime


class PermanentMemory:
    """Persist facts across conversations in a local JSON file."""

    def __init__(self, path: str = "memory_store.json"):
        self._path = Path(path)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2))

    def remember(self, key: str, value: str) -> str:
        self._data[key] = value
        self._save()
        return f"Stored: {key!r} = {value!r}"

    def recall(self, key: str) -> str:
        if key not in self._data:
            return f"No memory found for key: {key!r}"
        return self._data[key]

    def forget(self, key: str) -> str:
        if key not in self._data:
            return f"Key {key!r} not found."
        del self._data[key]
        self._save()
        return f"Forgotten: {key!r}"

    def list_memories(self) -> str:
        if not self._data:
            return "No memories stored."
        lines = [f"  {k!r}: {v!r}" for k, v in self._data.items()]
        return "Stored memories:\n" + "\n".join(lines)

    def as_context_string(self) -> str:
        """Return a compact string to inject into the system prompt."""
        if not self._data:
            return ""
        items = "; ".join(f"{k}={v}" for k, v in self._data.items())
        return f"[Persistent memory: {items}]"


class TemporaryMemory:
    """Conversation history for the current session."""

    def __init__(self):
        self._messages: list[dict] = []
        self._session_start = datetime.now().isoformat()

    def add(self, role: str, content: str | list) -> None:
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def summary(self) -> str:
        turns = sum(1 for m in self._messages if m["role"] == "user")
        return f"Session started {self._session_start} — {turns} user turn(s), {len(self._messages)} total messages."

"""Thread-safe state storage for the latest BCIState.

Uses threading.Lock for safe concurrent access between the BrainFlow
reader thread and the FastAPI server thread.
"""

import threading
import time
from typing import Any


class StateManager:
    """Stores the latest BCIState dict with thread-safe access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._last_update_ms: int | None = None
        self._session_id: str | None = None
        self._device_connected: bool = False

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the stored state (called from BrainFlow reader thread).

        Args:
            state: A BCIState dict conforming to bci_state.schema.json.
        """
        with self._lock:
            self._state = state
            self._last_update_ms = int(time.time() * 1000)
            self._session_id = state.get("session_id")

    def get_state(self) -> dict[str, Any] | None:
        """Get the latest state with staleness_ms updated.

        Returns:
            A copy of the BCIState dict with updated staleness_ms, or None if no state.
        """
        with self._lock:
            if self._state is None:
                return None
            now_ms = int(time.time() * 1000)
            state_copy = dict(self._state)
            state_copy["staleness_ms"] = now_ms - self._last_update_ms
            return state_copy

    @property
    def last_update_ms(self) -> int | None:
        with self._lock:
            return self._last_update_ms

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def device_connected(self) -> bool:
        with self._lock:
            return self._device_connected

    @device_connected.setter
    def device_connected(self, value: bool) -> None:
        with self._lock:
            self._device_connected = value

"""Thread-safe state storage for current BCI state.

Uses threading.Lock for safe concurrent access between the
BrainFlow reader thread and the FastAPI server thread.
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

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the stored BCI state (called from reader thread).

        Args:
            state: A dict conforming to bci_state.schema.json.
        """
        with self._lock:
            self._state = state
            self._last_update_ms = int(time.time() * 1000)

    def get_state(self) -> dict[str, Any] | None:
        """Get the latest BCI state (called from server thread).

        Returns:
            The latest BCIState dict, or None if no data has been received.
            The staleness_ms field is updated to reflect current time.
        """
        with self._lock:
            if self._state is None:
                return None
            state = dict(self._state)
            now_ms = int(time.time() * 1000)
            state["staleness_ms"] = now_ms - self._last_update_ms
            return state

    @property
    def last_update_ms(self) -> int | None:
        """Timestamp of the last state update, or None."""
        with self._lock:
            return self._last_update_ms

    @property
    def has_data(self) -> bool:
        """Whether any state data has been received."""
        with self._lock:
            return self._state is not None

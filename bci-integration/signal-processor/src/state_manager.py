"""Thread-safe state storage for the latest BCIState.

Uses threading.Lock for safe concurrent access between the BrainFlow
reader thread and the FastAPI server thread.
"""

import threading
import time
from typing import Any


class StateManager:
    """Stores the latest BCIState dict with thread-safe access."""

    def __init__(
        self,
        on_state_change: Any | None = None,
        on_state_update: Any | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._last_update_ms: int | None = None
        self._session_id: str | None = None
        self._device_connected: bool = False
        self._previous_primary: str | None = None
        self._on_state_change = on_state_change
        self._on_state_update = on_state_update

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the stored state (called from BrainFlow reader thread).

        Args:
            state: A BCIState dict conforming to bci_state.schema.json.
        """
        fire_callback = False
        old_state_copy = None
        new_state_copy = None

        with self._lock:
            old_primary = self._previous_primary

            # Capture old state BEFORE overwriting so callbacks get actual old data
            if self._state is not None:
                old_state_copy = dict(self._state)
                # Deep copy nested dicts to avoid shared references
                if "state" in old_state_copy:
                    old_state_copy["state"] = dict(old_state_copy["state"])
                if "scores" in old_state_copy:
                    old_state_copy["scores"] = dict(old_state_copy["scores"])
                if "band_powers" in old_state_copy and old_state_copy["band_powers"]:
                    old_state_copy["band_powers"] = dict(old_state_copy["band_powers"])

            new_primary = state.get("state", {}).get("primary")
            self._state = state
            self._last_update_ms = int(time.time() * 1000)
            self._session_id = state.get("session_id")
            self._previous_primary = new_primary

            if (
                old_primary is not None
                and new_primary is not None
                and old_primary != new_primary
                and self._on_state_change is not None
            ):
                fire_callback = True
                new_state_copy = dict(state)

        # Fire callback outside lock to avoid deadlocks
        if fire_callback:
            self._on_state_change(old_state_copy, new_state_copy)

        # Fire on_state_update on every call (not just transitions)
        if self._on_state_update is not None:
            self._on_state_update(dict(state))

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

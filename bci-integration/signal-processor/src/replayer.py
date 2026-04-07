"""Session replayer that reads a JSONL recording and feeds it to StateManager.

Preserves original timing between states.
"""

import json
import logging
import threading
import time
from typing import Any

from .state_manager import StateManager

logger = logging.getLogger(__name__)


class SessionReplayer:
    """Replays a recorded JSONL session through a StateManager.

    Args:
        file_path: Path to the JSONL recording file.
        state_manager: StateManager to feed replayed states into.
    """

    def __init__(self, file_path: str, state_manager: StateManager) -> None:
        self._file_path = file_path
        self._state_manager = state_manager
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._done = False
        self._total_lines = 0

    def start(self) -> None:
        """Launch the replay background thread."""
        # Count total lines for progress logging
        with open(self._file_path, "r", encoding="utf-8") as f:
            self._total_lines = sum(1 for line in f if line.strip())

        if self._total_lines == 0:
            logger.warning("Recording file %s is empty, nothing to replay", self._file_path)
            self._done = True
            return

        logger.info("Replaying %d states from %s", self._total_lines, self._file_path)
        self._state_manager.device_connected = True
        self._done = False
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._replay_loop, daemon=True, name="bci-replayer"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the replay thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._state_manager.device_connected = False
        logger.info("Replay stopped.")

    @property
    def is_done(self) -> bool:
        return self._done

    def _replay_loop(self) -> None:
        """Read JSONL line by line, preserving original timing."""
        prev_ts: int | None = None
        count = 0

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if self._stop_event.is_set():
                        break

                    line = line.strip()
                    if not line:
                        continue

                    state: dict[str, Any] = json.loads(line)
                    current_ts = state.get("timestamp_unix_ms", 0)

                    # Sleep for the delta between consecutive timestamps
                    if prev_ts is not None and current_ts > prev_ts:
                        delta_s = (current_ts - prev_ts) / 1000.0
                        # Cap individual sleep to 5s to avoid hanging on bad data
                        delta_s = min(delta_s, 5.0)
                        # Use stop_event.wait so we can be interrupted
                        if self._stop_event.wait(timeout=delta_s):
                            break

                    self._state_manager.update_state(state)
                    prev_ts = current_ts
                    count += 1

                    # Log progress every 10 states or on the last one
                    if count % 10 == 0 or count == self._total_lines:
                        logger.info(
                            "Replaying state %d of %d...", count, self._total_lines
                        )

        except Exception:
            logger.error("Error during replay", exc_info=True)
        finally:
            self._done = True
            self._state_manager.device_connected = False
            logger.info("Replay complete. %d states replayed.", count)

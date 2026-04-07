"""Session recorder that writes BCIState dicts as JSONL.

Thread-safe: designed to be called from the BrainFlow acquisition loop thread.
"""

import json
import logging
import threading
from typing import Any, TextIO

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records BCIState dicts to a JSONL file.

    Args:
        file_path: Path to the output JSONL file.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._lines_recorded = 0

    def start(self) -> None:
        """Open the file for writing."""
        with self._lock:
            if self._file is not None:
                raise RuntimeError("Recorder is already started")
            self._file = open(self._file_path, "w", encoding="utf-8")
            self._lines_recorded = 0
            logger.info("Recording session to %s", self._file_path)

    def record(self, bci_state: dict[str, Any]) -> None:
        """Append one BCIState dict as a JSON line.

        Args:
            bci_state: A BCIState dict (already contains timestamp_unix_ms).
        """
        with self._lock:
            if self._file is None:
                return
            line = json.dumps(bci_state, separators=(",", ":"))
            self._file.write(line + "\n")
            self._file.flush()
            self._lines_recorded += 1

    def stop(self) -> int:
        """Close the file and return the number of lines recorded."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                logger.info(
                    "Recording stopped. %d states written to %s",
                    self._lines_recorded,
                    self._file_path,
                )
            return self._lines_recorded

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._file is not None

    @property
    def lines_recorded(self) -> int:
        with self._lock:
            return self._lines_recorded

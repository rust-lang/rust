"""Tests for session recording and replay."""

import json
import os
import tempfile
import time

import pytest

from src.recorder import SessionRecorder
from src.replayer import SessionReplayer
from src.state_manager import StateManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_SESSION = os.path.join(FIXTURES_DIR, "sample_session.jsonl")


def _make_state(timestamp_ms: int, primary: str = "focused", confidence: float = 0.8) -> dict:
    """Build a minimal valid BCIState dict for testing."""
    return {
        "timestamp_unix_ms": timestamp_ms,
        "session_id": "session-test",
        "device_id": "test-device",
        "state": {"primary": primary, "confidence": confidence, "secondary": []},
        "scores": {"attention": 0.7, "relaxation": 0.3, "cognitive_load": 0.5},
        "band_powers": {
            "delta": 0.0001,
            "theta": 0.00008,
            "alpha": 0.00005,
            "beta": 0.0002,
            "gamma": 0.0001,
        },
        "signal_quality": 0.9,
        "artifact_probability": 0.05,
        "staleness_ms": 0,
        "natural_language_summary": f"User brain state: {primary.upper()} (confidence: {confidence:.2f})",
    }


class TestSessionRecorder:
    def test_record_writes_valid_jsonl(self):
        """Record a few states and verify the output is valid JSONL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            path = tmp.name

        try:
            recorder = SessionRecorder(file_path=path)
            recorder.start()
            assert recorder.is_recording

            states = [
                _make_state(1000, "focused"),
                _make_state(1250, "relaxed"),
                _make_state(1500, "drowsy"),
            ]
            for s in states:
                recorder.record(s)

            assert recorder.lines_recorded == 3
            count = recorder.stop()
            assert count == 3
            assert not recorder.is_recording

            # Verify file contents
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            assert len(lines) == 3
            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert parsed["timestamp_unix_ms"] == states[i]["timestamp_unix_ms"]
                assert parsed["state"]["primary"] == states[i]["state"]["primary"]
        finally:
            os.unlink(path)

    def test_record_before_start_is_noop(self):
        """Calling record() before start() should silently do nothing."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = tmp.name

        try:
            recorder = SessionRecorder(file_path=path)
            recorder.record(_make_state(1000))
            assert recorder.lines_recorded == 0
        finally:
            os.unlink(path)

    def test_start_twice_raises(self):
        """Starting a recorder that is already started should raise."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = tmp.name

        try:
            recorder = SessionRecorder(file_path=path)
            recorder.start()
            with pytest.raises(RuntimeError):
                recorder.start()
            recorder.stop()
        finally:
            os.unlink(path)

    def test_flush_on_each_write(self):
        """Data should be readable from the file after each write (flush)."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = tmp.name

        try:
            recorder = SessionRecorder(file_path=path)
            recorder.start()
            recorder.record(_make_state(1000, "focused"))

            # Read file while recorder is still open
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert len(content.strip()) > 0
            parsed = json.loads(content.strip())
            assert parsed["state"]["primary"] == "focused"

            recorder.stop()
        finally:
            os.unlink(path)


class TestSessionReplayer:
    def test_replay_updates_state_manager(self):
        """Replay a file and verify states appear in StateManager."""
        sm = StateManager()
        replayer = SessionReplayer(file_path=SAMPLE_SESSION, state_manager=sm)

        replayer.start()
        assert sm.device_connected

        # Wait for replay to finish (5 states at 250ms = ~1s plus some margin)
        deadline = time.time() + 5.0
        while not replayer.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert replayer.is_done

        # The last state in the fixture is "drowsy"
        state = sm.get_state()
        assert state is not None
        assert state["state"]["primary"] == "drowsy"

    def test_replay_sets_device_connected(self):
        """Verify device_connected is True during replay and False after."""
        sm = StateManager()
        replayer = SessionReplayer(file_path=SAMPLE_SESSION, state_manager=sm)

        assert not sm.device_connected
        replayer.start()
        assert sm.device_connected

        deadline = time.time() + 5.0
        while not replayer.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert replayer.is_done
        # After replay finishes, device_connected should be False
        assert not sm.device_connected

    def test_replay_timing(self):
        """Verify replay roughly preserves original timing deltas."""
        # Create a file with 3 states spaced 200ms apart
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            path = tmp.name
            for i in range(3):
                state = _make_state(1000 + i * 200, "focused")
                tmp.write(json.dumps(state) + "\n")

        try:
            sm = StateManager()
            replayer = SessionReplayer(file_path=path, state_manager=sm)

            t0 = time.time()
            replayer.start()

            deadline = time.time() + 5.0
            while not replayer.is_done and time.time() < deadline:
                time.sleep(0.05)

            elapsed = time.time() - t0
            # 3 states with 200ms gaps = 400ms total delay
            # Allow generous tolerance (100ms - 1500ms)
            assert 0.1 <= elapsed <= 1.5, f"Replay took {elapsed:.3f}s, expected ~0.4s"
        finally:
            os.unlink(path)

    def test_replay_empty_file(self):
        """Replaying an empty file should immediately be done."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            path = tmp.name

        try:
            sm = StateManager()
            replayer = SessionReplayer(file_path=path, state_manager=sm)
            replayer.start()
            assert replayer.is_done
        finally:
            os.unlink(path)

    def test_replay_stop_interrupts(self):
        """Calling stop() should interrupt an ongoing replay."""
        # Create a file with many states spaced far apart
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            path = tmp.name
            for i in range(20):
                state = _make_state(1000 + i * 1000, "focused")  # 1s gaps
                tmp.write(json.dumps(state) + "\n")

        try:
            sm = StateManager()
            replayer = SessionReplayer(file_path=path, state_manager=sm)
            replayer.start()

            # Let it start, then stop
            time.sleep(0.2)
            replayer.stop()

            # Should have stopped before finishing all 20 states
            state = sm.get_state()
            # It should have processed at least 1 state
            assert state is not None
        finally:
            os.unlink(path)


class TestRoundTrip:
    def test_record_then_replay(self):
        """Record states, then replay them and compare."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = tmp.name

        try:
            # Record
            original_states = [
                _make_state(1000, "focused", 0.8),
                _make_state(1250, "relaxed", 0.7),
                _make_state(1500, "drowsy", 0.6),
            ]

            recorder = SessionRecorder(file_path=path)
            recorder.start()
            for s in original_states:
                recorder.record(s)
            recorder.stop()

            # Replay
            sm = StateManager()
            replayer = SessionReplayer(file_path=path, state_manager=sm)
            replayer.start()

            deadline = time.time() + 5.0
            while not replayer.is_done and time.time() < deadline:
                time.sleep(0.05)

            assert replayer.is_done

            # Verify the final state matches the last recorded state
            state = sm.get_state()
            assert state is not None
            assert state["state"]["primary"] == "drowsy"
            assert state["state"]["confidence"] == 0.6
            assert state["session_id"] == "session-test"

            # Verify recorded file matches original states
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == len(original_states)
            for recorded, original in zip(lines, original_states):
                assert recorded["timestamp_unix_ms"] == original["timestamp_unix_ms"]
                assert recorded["state"]["primary"] == original["state"]["primary"]
                assert recorded["scores"] == original["scores"]
        finally:
            os.unlink(path)

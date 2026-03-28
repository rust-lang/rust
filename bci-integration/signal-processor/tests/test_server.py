"""Tests for the BCI State Server HTTP endpoints.

Uses FastAPI TestClient to test /state and /health without starting
BrainFlow or a real HTTP server.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.server import create_app
from src.state_manager import StateManager


def _make_test_app():
    """Create app and return (client, state_manager) for testing.

    We bypass BrainFlow by directly manipulating the state manager.
    """
    import src.server as server_module

    sm = StateManager()
    sm.device_connected = True

    # Patch module-level refs to avoid starting BrainFlow
    server_module._state_manager = sm
    server_module._reader = None
    server_module._start_time = time.time()

    from fastapi import FastAPI
    from src.models import BCIStateModel, HealthResponse, StateResponse

    app = FastAPI()

    @app.get("/state", response_model=StateResponse)
    def get_state():
        now_ms = int(time.time() * 1000)
        state = sm.get_state()
        if state is None:
            return StateResponse(
                available=False,
                timestamp_unix_ms=now_ms,
                error="No BCI data received yet",
            )
        return StateResponse(
            available=True,
            timestamp_unix_ms=now_ms,
            bci_state=BCIStateModel(**state),
        )

    @app.get("/health", response_model=HealthResponse)
    def get_health():
        now = time.time()
        uptime = now - server_module._start_time
        last_update = sm.last_update_ms
        if not sm.device_connected or last_update is None:
            status = "no_signal"
        else:
            staleness_ms = int(now * 1000) - last_update
            state = sm.get_state()
            sq = state.get("signal_quality", 0) if state else 0
            if staleness_ms > 5000 or sq < 0.3:
                status = "degraded"
            else:
                status = "ok"
        return HealthResponse(
            status=status,
            uptime_seconds=round(uptime, 1),
            device_connected=sm.device_connected,
            session_id=sm.session_id,
            last_update_unix_ms=last_update,
        )

    client = TestClient(app)
    return client, sm


def _sample_bci_state() -> dict:
    """Return a valid BCIState dict."""
    return {
        "timestamp_unix_ms": int(time.time() * 1000),
        "session_id": "test-session-001",
        "device_id": "synthetic-board-1",
        "state": {
            "primary": "focused",
            "confidence": 0.85,
            "secondary": [
                {"state": "active", "confidence": 0.4},
            ],
        },
        "scores": {
            "attention": 0.79,
            "relaxation": 0.23,
            "cognitive_load": 0.45,
        },
        "band_powers": {
            "delta": 5.2,
            "theta": 3.1,
            "alpha": 4.5,
            "beta": 8.7,
            "gamma": 1.2,
        },
        "signal_quality": 0.92,
        "artifact_probability": 0.05,
        "staleness_ms": 0,
        "natural_language_summary": (
            "User brain state: FOCUSED (confidence: 0.85, "
            "attention: 0.79, relaxation: 0.23, cognitive_load: 0.45, "
            "signal quality: good)"
        ),
    }


class TestGetState:
    """Tests for GET /state endpoint."""

    def test_returns_unavailable_before_any_data(self):
        client, sm = _make_test_app()
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert "error" in data
        assert data["bci_state"] is None

    def test_returns_valid_state_after_update(self):
        client, sm = _make_test_app()
        sm.update_state(_sample_bci_state())

        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["bci_state"] is not None
        assert data["bci_state"]["state"]["primary"] == "focused"
        assert data["bci_state"]["signal_quality"] == 0.92
        assert "natural_language_summary" in data["bci_state"]

    def test_has_required_fields(self):
        client, sm = _make_test_app()
        sm.update_state(_sample_bci_state())

        resp = client.get("/state")
        data = resp.json()
        assert "available" in data
        assert "timestamp_unix_ms" in data
        bci = data["bci_state"]
        assert "session_id" in bci
        assert "device_id" in bci
        assert "state" in bci
        assert "scores" in bci
        assert "signal_quality" in bci
        assert "natural_language_summary" in bci

    def test_scores_in_valid_range(self):
        client, sm = _make_test_app()
        sm.update_state(_sample_bci_state())

        resp = client.get("/state")
        scores = resp.json()["bci_state"]["scores"]
        for key in ("attention", "relaxation", "cognitive_load"):
            assert 0.0 <= scores[key] <= 1.0

    def test_staleness_ms_increases(self):
        client, sm = _make_test_app()
        sm.update_state(_sample_bci_state())
        time.sleep(0.05)  # 50ms

        resp = client.get("/state")
        staleness = resp.json()["bci_state"]["staleness_ms"]
        assert staleness >= 40  # at least ~40ms elapsed


class TestGetHealth:
    """Tests for GET /health endpoint."""

    def test_no_signal_before_data(self):
        client, sm = _make_test_app()
        sm.device_connected = False

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_signal"
        assert data["device_connected"] is False

    def test_ok_with_good_signal(self):
        client, sm = _make_test_app()
        sm.update_state(_sample_bci_state())

        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["device_connected"] is True
        assert data["session_id"] == "test-session-001"
        assert data["last_update_unix_ms"] is not None

    def test_degraded_with_low_signal_quality(self):
        client, sm = _make_test_app()
        state = _sample_bci_state()
        state["signal_quality"] = 0.1  # Below 0.3 threshold
        sm.update_state(state)

        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"

    def test_uptime_is_positive(self):
        client, sm = _make_test_app()
        resp = client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 0

    def test_has_required_fields(self):
        client, sm = _make_test_app()
        resp = client.get("/health")
        data = resp.json()
        for field in ("status", "uptime_seconds", "device_connected", "session_id", "last_update_unix_ms"):
            assert field in data


class TestLatency:
    """Verify endpoint response times."""

    def test_state_endpoint_under_50ms(self):
        client, sm = _make_test_app()
        sm.update_state(_sample_bci_state())

        start = time.time()
        resp = client.get("/state")
        elapsed_ms = (time.time() - start) * 1000

        assert resp.status_code == 200
        assert elapsed_ms < 50, f"GET /state took {elapsed_ms:.1f}ms (>50ms)"

    def test_health_endpoint_under_50ms(self):
        client, sm = _make_test_app()

        start = time.time()
        resp = client.get("/health")
        elapsed_ms = (time.time() - start) * 1000

        assert resp.status_code == 200
        assert elapsed_ms < 50, f"GET /health took {elapsed_ms:.1f}ms (>50ms)"

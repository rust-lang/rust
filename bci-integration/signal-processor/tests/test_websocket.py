"""Tests for WebSocket streaming of BCI state updates.

Uses FastAPI TestClient's WebSocket support to verify real-time
state broadcasting, connection management, and the /ws/info endpoint.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.server import create_app
from src.state_manager import StateManager
from src.ws_manager import WebSocketManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_bci_state(
    primary: str = "focused",
    attention: float = 0.79,
    relaxation: float = 0.23,
    cognitive_load: float = 0.45,
) -> dict[str, Any]:
    """Return a valid BCIState dict with customisable fields."""
    return {
        "timestamp_unix_ms": int(time.time() * 1000),
        "session_id": "test-session-001",
        "device_id": "synthetic-board-1",
        "state": {
            "primary": primary,
            "confidence": 0.85,
            "secondary": [],
        },
        "scores": {
            "attention": attention,
            "relaxation": relaxation,
            "cognitive_load": cognitive_load,
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
        "natural_language_summary": f"User brain state: {primary.upper()}",
    }


def _make_test_app() -> tuple[TestClient, StateManager, WebSocketManager]:
    """Create app via create_app with a pre-built StateManager."""
    sm = StateManager()
    sm.device_connected = True
    app = create_app(synthetic=True, state_manager=sm)

    import src.server as server_mod

    ws_mgr = server_mod._ws_manager

    client = TestClient(app)
    return client, sm, ws_mgr


# ---------------------------------------------------------------------------
# WebSocket endpoint tests
# ---------------------------------------------------------------------------


class TestWebSocketConnect:
    """Tests for WS /ws endpoint -- connection and initial state."""

    def test_connect_receives_initial_state(self):
        """Client should receive the current state immediately on connect."""
        client, sm, ws_mgr = _make_test_app()
        sm.update_state(_sample_bci_state(primary="focused"))

        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["state"]["primary"] == "focused"
            assert data["session_id"] == "test-session-001"

    def test_connect_no_initial_state_when_empty(self):
        """If no state exists yet, the connection should still succeed
        without sending an initial message."""
        client, sm, ws_mgr = _make_test_app()

        # We connect; since sm has no data, no initial state is sent.
        # Verify by checking the connection is established and alive.
        with client.websocket_connect("/ws") as ws:
            assert ws_mgr.connection_count >= 1

    def test_connect_after_state_exists_gets_current_state(self):
        """A client connecting after state is already available gets it
        immediately."""
        client, sm, ws_mgr = _make_test_app()

        # Set state before connecting
        sm.update_state(_sample_bci_state(primary="meditative"))

        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["state"]["primary"] == "meditative"
            assert data["signal_quality"] == 0.92


class TestWebSocketDisconnect:
    """Tests for disconnect handling -- no crashes."""

    def test_disconnect_removes_client(self):
        """After a client disconnects, connection_count should drop."""
        client, sm, ws_mgr = _make_test_app()

        with client.websocket_connect("/ws"):
            assert ws_mgr.connection_count >= 1

        # After exiting the context manager, the connection is closed.
        assert ws_mgr.connection_count == 0

    def test_broadcast_after_disconnect_does_not_crash(self):
        """Broadcasting after a client disconnects should not raise."""
        client, sm, ws_mgr = _make_test_app()

        with client.websocket_connect("/ws"):
            pass  # connect and immediately disconnect

        # Push an update -- broadcast to zero clients should be fine.
        # The on_state_update callback may fail silently (no event loop),
        # but calling broadcast directly should work.
        sm.update_state(_sample_bci_state(primary="active"))
        # No exception means success


# ---------------------------------------------------------------------------
# /ws/info endpoint tests
# ---------------------------------------------------------------------------


class TestWsInfo:
    """Tests for GET /ws/info monitoring endpoint."""

    def test_returns_zero_connections_initially(self):
        client, sm, ws_mgr = _make_test_app()
        resp = client.get("/ws/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connections"] == 0

    def test_returns_correct_count_with_one_connection(self):
        client, sm, ws_mgr = _make_test_app()

        with client.websocket_connect("/ws"):
            resp = client.get("/ws/info")
            assert resp.json()["connections"] >= 1

    def test_returns_zero_after_disconnect(self):
        client, sm, ws_mgr = _make_test_app()

        with client.websocket_connect("/ws"):
            pass

        resp = client.get("/ws/info")
        assert resp.json()["connections"] == 0

    def test_returns_correct_count_with_multiple_connections(self):
        client, sm, ws_mgr = _make_test_app()

        with client.websocket_connect("/ws"):
            with client.websocket_connect("/ws"):
                resp = client.get("/ws/info")
                assert resp.json()["connections"] >= 2


# ---------------------------------------------------------------------------
# WebSocketManager unit tests
# ---------------------------------------------------------------------------


class TestWebSocketManagerUnit:
    """Direct unit tests on WebSocketManager without FastAPI."""

    def test_connection_count_starts_at_zero(self):
        mgr = WebSocketManager()
        assert mgr.connection_count == 0

    def test_disconnect_nonexistent_is_safe(self):
        """Calling disconnect with an unknown ws should not raise."""
        mgr = WebSocketManager()

        class FakeWS:
            pass

        mgr.disconnect(FakeWS())  # type: ignore
        assert mgr.connection_count == 0

    def test_broadcast_empty_connections(self):
        """Broadcast with no connections should be a no-op."""
        mgr = WebSocketManager()
        asyncio.run(mgr.broadcast({"state": {"primary": "focused"}}))
        assert mgr.connection_count == 0

    def test_broadcast_sends_to_all_connections(self):
        """Broadcast should send serialized JSON to every connection."""
        mgr = WebSocketManager()

        # Create mock WebSocket objects
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        # Manually add them (bypassing accept())
        mgr._connections.add(ws1)
        mgr._connections.add(ws2)
        assert mgr.connection_count == 2

        state = {"state": {"primary": "focused"}, "scores": {"attention": 0.8}}
        asyncio.run(mgr.broadcast(state))

        # Both should have received the same JSON text
        expected = json.dumps(state)
        ws1.send_text.assert_awaited_once_with(expected)
        ws2.send_text.assert_awaited_once_with(expected)

    def test_broadcast_removes_dead_connections(self):
        """If a connection raises on send, it should be removed."""
        mgr = WebSocketManager()

        alive = AsyncMock()
        dead = AsyncMock()
        dead.send_text.side_effect = Exception("connection closed")

        mgr._connections.add(alive)
        mgr._connections.add(dead)
        assert mgr.connection_count == 2

        asyncio.run(mgr.broadcast({"state": {"primary": "active"}}))

        # Dead connection should have been removed
        assert mgr.connection_count == 1
        assert dead not in mgr._connections
        assert alive in mgr._connections

    def test_broadcast_serializes_once(self):
        """JSON should be serialized once and the same string sent to all."""
        mgr = WebSocketManager()

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        mgr._connections.update({ws1, ws2, ws3})

        state = {"key": "value"}
        asyncio.run(mgr.broadcast(state))

        # All three should receive identical string
        calls = [
            ws1.send_text.call_args[0][0],
            ws2.send_text.call_args[0][0],
            ws3.send_text.call_args[0][0],
        ]
        assert all(c == calls[0] for c in calls)
        assert json.loads(calls[0]) == state


# ---------------------------------------------------------------------------
# StateManager on_state_update callback tests
# ---------------------------------------------------------------------------


class TestStateManagerOnStateUpdate:
    """Verify that on_state_update fires on every update_state call."""

    def test_fires_on_every_update(self):
        """on_state_update should be called for every update_state, not
        just state transitions."""
        updates = []
        sm = StateManager(on_state_update=lambda s: updates.append(s))
        sm.device_connected = True

        sm.update_state(_sample_bci_state(primary="focused"))
        sm.update_state(_sample_bci_state(primary="focused"))  # same state
        sm.update_state(_sample_bci_state(primary="relaxed"))  # transition

        assert len(updates) == 3
        assert updates[0]["state"]["primary"] == "focused"
        assert updates[1]["state"]["primary"] == "focused"
        assert updates[2]["state"]["primary"] == "relaxed"

    def test_fires_independently_of_on_state_change(self):
        """on_state_update fires on every call; on_state_change only on
        transitions."""
        changes = []
        updates = []
        sm = StateManager(
            on_state_change=lambda old, new: changes.append((old, new)),
            on_state_update=lambda s: updates.append(s),
        )

        sm.update_state(_sample_bci_state(primary="focused"))
        sm.update_state(_sample_bci_state(primary="focused"))
        sm.update_state(_sample_bci_state(primary="relaxed"))
        sm.update_state(_sample_bci_state(primary="relaxed"))

        assert len(updates) == 4  # every call
        assert len(changes) == 1  # only focused -> relaxed

    def test_no_callback_when_none(self):
        """If on_state_update is None, update_state should still work."""
        sm = StateManager(on_state_update=None)
        sm.update_state(_sample_bci_state(primary="focused"))
        assert sm.get_state() is not None

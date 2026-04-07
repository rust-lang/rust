"""Tests for webhook registration, delivery, filtering, and cooldown.

Uses FastAPI TestClient for the HTTP endpoints and a local HTTP server
to receive webhook deliveries.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.server import create_app
from src.state_manager import StateManager
from src.webhook_manager import WebhookManager


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


class _WebhookReceiver:
    """Lightweight HTTP server that captures POST payloads."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}/hook"

    def start(self) -> None:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                import json

                with receiver._lock:
                    receiver.payloads.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):  # silence logs
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def wait_for_payloads(self, count: int = 1, timeout: float = 3.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.payloads) >= count:
                    return list(self.payloads)
            time.sleep(0.05)
        with self._lock:
            return list(self.payloads)


def _make_test_app() -> tuple[TestClient, StateManager, WebhookManager]:
    """Create app via create_app with a pre-built StateManager."""
    sm = StateManager()
    sm.device_connected = True
    app = create_app(synthetic=True, state_manager=sm)

    # Grab the webhook_manager that create_app attached
    import src.server as server_mod

    wm = server_mod._webhook_manager

    client = TestClient(app)
    return client, sm, wm


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestWebhookEndpoints:
    """Test POST/GET/DELETE /webhooks endpoints."""

    def test_register_and_list(self):
        client, sm, wm = _make_test_app()
        resp = client.post("/webhooks", json={"url": "http://example.com/hook"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["registered"] is True
        assert "id" in data

        resp = client.get("/webhooks")
        assert resp.status_code == 200
        hooks = resp.json()
        assert len(hooks) == 1
        assert hooks[0]["url"] == "http://example.com/hook"
        assert hooks[0]["id"] == data["id"]

    def test_register_with_filters(self):
        client, sm, wm = _make_test_app()
        resp = client.post(
            "/webhooks",
            json={
                "url": "http://example.com/hook",
                "filters": {"state_is": "relaxed", "score_above": {"attention": 0.5}},
            },
        )
        assert resp.status_code == 200

        hooks = client.get("/webhooks").json()
        assert hooks[0]["filters"]["state_is"] == "relaxed"
        assert hooks[0]["filters"]["score_above"] == {"attention": 0.5}

    def test_delete_webhook(self):
        client, sm, wm = _make_test_app()
        wh_id = client.post(
            "/webhooks", json={"url": "http://example.com/hook"}
        ).json()["id"]

        resp = client.delete(f"/webhooks/{wh_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        hooks = client.get("/webhooks").json()
        assert len(hooks) == 0

    def test_delete_nonexistent_returns_404(self):
        client, sm, wm = _make_test_app()
        resp = client.delete("/webhooks/no-such-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Webhook delivery tests
# ---------------------------------------------------------------------------


class TestWebhookDelivery:
    """Test that state transitions fire webhooks correctly."""

    def test_fires_on_state_change(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(url=receiver.url)

            # First state (establishes baseline)
            sm.update_state(_sample_bci_state(primary="focused"))
            # Second state (triggers transition)
            sm.update_state(_sample_bci_state(primary="relaxed"))

            payloads = receiver.wait_for_payloads(1)
            assert len(payloads) == 1
            assert payloads[0]["event"] == "state_change"
            assert payloads[0]["previous_state"] == "focused"
            assert payloads[0]["new_state"] == "relaxed"
            assert "bci_state" in payloads[0]
            assert "timestamp_unix_ms" in payloads[0]
        finally:
            wm.shutdown()
            receiver.stop()

    def test_no_fire_when_state_unchanged(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(url=receiver.url)

            sm.update_state(_sample_bci_state(primary="focused"))
            sm.update_state(_sample_bci_state(primary="focused"))

            payloads = receiver.wait_for_payloads(1, timeout=0.5)
            assert len(payloads) == 0
        finally:
            wm.shutdown()
            receiver.stop()

    def test_cooldown_prevents_rapid_firing(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(url=receiver.url)

            # Transition 1: focused -> relaxed
            sm.update_state(_sample_bci_state(primary="focused"))
            sm.update_state(_sample_bci_state(primary="relaxed"))

            # Wait for delivery
            receiver.wait_for_payloads(1)

            # Transition 2 immediately: relaxed -> active (within cooldown)
            sm.update_state(_sample_bci_state(primary="active"))

            # Should still only have 1 payload due to cooldown
            payloads = receiver.wait_for_payloads(2, timeout=0.5)
            assert len(payloads) == 1
        finally:
            wm.shutdown()
            receiver.stop()

    def test_filter_state_is_matches(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(url=receiver.url, filters={"state_is": "relaxed"})

            sm.update_state(_sample_bci_state(primary="focused"))
            sm.update_state(_sample_bci_state(primary="relaxed"))

            payloads = receiver.wait_for_payloads(1)
            assert len(payloads) == 1
            assert payloads[0]["new_state"] == "relaxed"
        finally:
            wm.shutdown()
            receiver.stop()

    def test_filter_state_is_no_match(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            # Only fires when new state is "meditative"
            wm.register(url=receiver.url, filters={"state_is": "meditative"})

            sm.update_state(_sample_bci_state(primary="focused"))
            sm.update_state(_sample_bci_state(primary="relaxed"))

            payloads = receiver.wait_for_payloads(1, timeout=0.5)
            assert len(payloads) == 0
        finally:
            wm.shutdown()
            receiver.stop()

    def test_filter_score_below(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(
                url=receiver.url, filters={"score_below": {"attention": 0.5}}
            )

            sm.update_state(_sample_bci_state(primary="focused", attention=0.8))
            # Transition with low attention
            sm.update_state(_sample_bci_state(primary="relaxed", attention=0.3))

            payloads = receiver.wait_for_payloads(1)
            assert len(payloads) == 1
        finally:
            wm.shutdown()
            receiver.stop()

    def test_filter_score_below_no_match(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(
                url=receiver.url, filters={"score_below": {"attention": 0.5}}
            )

            sm.update_state(_sample_bci_state(primary="focused", attention=0.8))
            # Transition but attention is still above threshold
            sm.update_state(_sample_bci_state(primary="relaxed", attention=0.7))

            payloads = receiver.wait_for_payloads(1, timeout=0.5)
            assert len(payloads) == 0
        finally:
            wm.shutdown()
            receiver.stop()

    def test_filter_score_above(self):
        receiver = _WebhookReceiver()
        receiver.start()
        try:
            client, sm, wm = _make_test_app()
            wm.register(
                url=receiver.url, filters={"score_above": {"relaxation": 0.6}}
            )

            sm.update_state(_sample_bci_state(primary="focused", relaxation=0.2))
            sm.update_state(_sample_bci_state(primary="relaxed", relaxation=0.8))

            payloads = receiver.wait_for_payloads(1)
            assert len(payloads) == 1
        finally:
            wm.shutdown()
            receiver.stop()


# ---------------------------------------------------------------------------
# WebhookManager unit tests
# ---------------------------------------------------------------------------


class TestWebhookManagerUnit:
    """Direct unit tests on WebhookManager without HTTP."""

    def test_register_returns_id(self):
        wm = WebhookManager()
        wh_id = wm.register("http://example.com/hook")
        assert isinstance(wh_id, str)
        assert len(wh_id) > 0
        wm.shutdown()

    def test_unregister_returns_false_for_unknown(self):
        wm = WebhookManager()
        assert wm.unregister("nonexistent") is False
        wm.shutdown()

    def test_list_returns_registered(self):
        wm = WebhookManager()
        wh_id = wm.register("http://example.com/hook", filters={"state_is": "relaxed"})
        hooks = wm.list()
        assert len(hooks) == 1
        assert hooks[0]["id"] == wh_id
        assert hooks[0]["url"] == "http://example.com/hook"
        assert hooks[0]["filters"] == {"state_is": "relaxed"}
        assert "created_at" in hooks[0]
        wm.shutdown()

    def test_unregister_removes_webhook(self):
        wm = WebhookManager()
        wh_id = wm.register("http://example.com/hook")
        assert wm.unregister(wh_id) is True
        assert len(wm.list()) == 0
        wm.shutdown()

    def test_check_and_fire_ignores_same_state(self):
        wm = WebhookManager()
        wm.register("http://127.0.0.1:1/hook")  # unreachable, but shouldn't fire
        old = _sample_bci_state(primary="focused")
        new = _sample_bci_state(primary="focused")
        # Should not raise or attempt delivery
        wm.check_and_fire(old, new)
        wm.shutdown()

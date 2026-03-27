# BCI-to-OpenClaw Integration Prototype -- Task Breakdown

## Dependency Graph

```
Task 1 (Scaffold)
  |
  +---> Task 2 (BCI Reader)
  |       |
  |       +---> Task 3 (Feature Extractor)
  |       |       |
  |       |       +---> Task 8 (Replay Mode)
  |       |
  +---> Task 4 (Model Backend)
  |       |
  |       +---> Task 5 (Gateway)
  |
  +---> Task 6 (Monitor UI)
  |
  Tasks 2-6 ---> Task 7 (Process Manager)
  |
  Tasks 7,8 ---> Task 9 (Integration Tests)
```

---

## Task 1: Project Scaffolding and Shared Infrastructure
**Branch:** `feature/project-scaffold`
**Depends on:** None
**Estimate:** 2h

**Description:** Set up the Python project structure, shared configuration, and dependency management. Create the top-level directory layout with one package per component, a shared utilities package for schema validation and ZMQ helpers, and a central configuration module.

**Files:**
- `bci-integration/pyproject.toml` -- pinned deps: brainflow, pyzmq, numpy, scipy, scikit-learn, flask, rich, jsonschema
- `bci-integration/src/__init__.py`
- `bci-integration/src/shared/__init__.py`
- `bci-integration/src/shared/config.py` -- ZMQ addresses, model URL, constants
- `bci-integration/src/shared/schemas.py` -- load JSON schemas, provide validation helpers
- `bci-integration/src/shared/zmq_helpers.py` -- PUB/SUB socket factory functions
- `bci-integration/src/shared/logging_setup.py` -- structured logging to stderr
- `bci-integration/src/{bci_reader,feature_extractor,gateway,model_backend,monitor_ui}/__init__.py`
- `bci-integration/tests/test_schemas.py`

**Acceptance criteria:**
- [ ] `pip install -e .` succeeds with all dependencies
- [ ] Config module exports all socket addresses and constants
- [ ] Schema validation helpers accept valid payloads and reject invalid ones
- [ ] All 4 schemas have at least one positive and one negative test case

---

## Task 2: BCI Reader -- Synthetic Mode
**Branch:** `feature/bci-reader`
**Depends on:** Task 1
**Estimate:** 3h

**Description:** Implement the BCI Reader using BrainFlow's synthetic board. Read samples at 250Hz, package as `BCIStreamPacket`, publish on ZMQ PUB socket. Include `--synthetic` flag, connection retry with backoff, graceful shutdown.

**Files:**
- `bci-integration/src/bci_reader/reader.py`
- `bci-integration/src/bci_reader/__main__.py`
- `bci-integration/tests/test_bci_reader.py`

**Acceptance criteria:**
- [ ] `python -m src.bci_reader --synthetic` publishes valid BCIStreamPacket JSON on tcp://localhost:5555
- [ ] Each message passes schema validation
- [ ] `packet_id` increments monotonically
- [ ] `channels.eeg.values_uv` has correct channel count
- [ ] Clean shutdown on SIGINT
- [ ] Retry logic on BrainFlow init failure (3 retries, exponential backoff)

---

## Task 3: Feature Extractor
**Branch:** `feature/feature-extractor`
**Depends on:** Task 1, Task 2
**Estimate:** 4h

**Description:** Subscribe to BCI Reader stream, buffer 1-second sliding windows, compute band powers via Welch's method, derive attention/relaxation/cognitive_load scores, estimate artifact probability, assess signal quality. Emit `ProcessedBCIFeatures` every 250ms.

**Files:**
- `bci-integration/src/feature_extractor/extractor.py`
- `bci-integration/src/feature_extractor/dsp.py` -- pure DSP functions
- `bci-integration/src/feature_extractor/__main__.py`
- `bci-integration/tests/test_dsp.py`
- `bci-integration/tests/test_feature_extractor.py`

**Acceptance criteria:**
- [ ] Emits ~4 messages/second conforming to `processed_features.schema.json`
- [ ] Band powers are non-negative
- [ ] 10Hz sine wave input -> alpha band dominant
- [ ] `source_packet_range` correctly reflects window boundaries
- [ ] Degrades gracefully on no data (signal_quality=0 after 2s)
- [ ] NaN/Inf replaced with 0, artifact_probability set to 1

---

## Task 4: Model Backend
**Branch:** `feature/model-backend`
**Depends on:** Task 1
**Estimate:** 2h

**Description:** Flask server with `POST /classify` and `GET /health`. Threshold-based heuristic classifier: high alpha = relaxed, high beta = focused, high theta = drowsy, etc. Return `OpenClawModelOutput` with brain_state.

**Files:**
- `bci-integration/src/model_backend/server.py`
- `bci-integration/src/model_backend/classifier.py`
- `bci-integration/src/model_backend/__main__.py`
- `bci-integration/tests/test_model_backend.py`
- `bci-integration/tests/test_classifier.py`

**Acceptance criteria:**
- [ ] `GET /health` returns 200
- [ ] Valid classify request returns 200 with valid `model_output.schema.json`
- [ ] Invalid input returns 400 with error message
- [ ] High-alpha -> "relaxed" with confidence > 0.5
- [ ] High-beta -> "focused" with confidence > 0.5
- [ ] `latency_ms` populated, inference under 50ms

---

## Task 5: OpenClaw Gateway
**Branch:** `feature/openclaw-gateway`
**Depends on:** Task 1, Task 4
**Estimate:** 3h

**Description:** Bridge BCI feature stream to model backend. Subscribe to features on tcp://localhost:5556, construct `OpenClawModelInput`, POST to model backend, publish `OpenClawModelOutput` on tcp://localhost:5557. Handle timeouts and errors.

**Files:**
- `bci-integration/src/gateway/gateway.py`
- `bci-integration/src/gateway/__main__.py`
- `bci-integration/tests/test_gateway.py`

**Acceptance criteria:**
- [ ] Subscribes on :5556, publishes on :5557
- [ ] Output conforms to `model_output.schema.json`
- [ ] Unique `request_id` per request
- [ ] Model reachable -> `status: "ok"` with valid brain_state
- [ ] Model unreachable -> `status: "error"`
- [ ] Model timeout (>100ms) -> `status: "timeout"`, `primary_state: "unknown"`
- [ ] Overhead under 5ms

---

## Task 6: Monitor UI
**Branch:** `feature/monitor-ui`
**Depends on:** Task 1
**Estimate:** 2h

**Description:** Terminal dashboard using `rich`. Subscribe to gateway output on tcp://localhost:5557. Display brain state, band powers, signal quality, latency metrics, state transition log.

**Files:**
- `bci-integration/src/monitor_ui/dashboard.py`
- `bci-integration/src/monitor_ui/__main__.py`
- `bci-integration/tests/test_monitor_ui.py`

**Acceptance criteria:**
- [ ] Shows current `primary_state` with confidence
- [ ] Band power horizontal bars
- [ ] Signal quality colored indicator
- [ ] Live latency metrics
- [ ] "Waiting for data..." when no input
- [ ] Scrolling state transition log (last 10)
- [ ] Clean shutdown on SIGINT

---

## Task 7: Process Manager and Run Script
**Branch:** `feature/process-manager`
**Depends on:** Task 2, Task 3, Task 4, Task 5, Task 6
**Estimate:** 2h

**Description:** Shell script to start/stop/status all 5 components in dependency order. Health check Model Backend before starting Gateway. Log output to `logs/`.

**Files:**
- `bci-integration/run.sh`
- `bci-integration/Procfile`

**Acceptance criteria:**
- [ ] `./run.sh start` launches all components in correct order
- [ ] `./run.sh stop` cleanly terminates all
- [ ] `./run.sh status` shows running PIDs
- [ ] Per-component log files in `logs/`
- [ ] Idempotent start (warns if already running)

---

## Task 8: Recording and Replay Mode
**Branch:** `feature/replay-mode`
**Depends on:** Task 2, Task 3
**Estimate:** 2h

**Description:** Add `--record` and `--replay` flags to BCI Reader. Record to JSONL, replay with original timing. Enables deterministic testing without hardware.

**Files:**
- `bci-integration/src/bci_reader/recorder.py`
- `bci-integration/src/bci_reader/replay.py`
- `bci-integration/tests/test_replay.py`
- `bci-integration/tests/fixtures/sample_session.jsonl`

**Acceptance criteria:**
- [ ] `--record session.jsonl` writes valid JSONL while publishing
- [ ] `--replay session.jsonl` publishes with original timing
- [ ] Replayed packets match recorded packets
- [ ] Clean exit when file exhausted

---

## Task 9: Integration Test and End-to-End Validation
**Branch:** `feature/integration-tests`
**Depends on:** Task 7, Task 8
**Estimate:** 3h

**Description:** End-to-end tests using replay mode. Verify schema conformance at every boundary, latency targets, and correct classification output.

**Files:**
- `bci-integration/tests/test_integration.py`
- `bci-integration/tests/test_latency.py`
- `bci-integration/tests/run_integration.sh`

**Acceptance criteria:**
- [ ] Full pipeline runs with replay data, output collected from :5557
- [ ] All messages at all boundaries pass schema validation
- [ ] End-to-end latency under 200ms for 95% of messages
- [ ] Model inference under 50ms for 100% of messages
- [ ] No hardware required
- [ ] `run_integration.sh` exits 0/non-zero appropriately

---

## Execution Order (Solo Developer)

| Phase | Tasks | Parallel? | Cumulative |
|-------|-------|-----------|------------|
| 1 | Task 1 (Scaffold) | -- | ~2h |
| 2 | Task 2 (BCI Reader) + Task 4 (Model Backend) + Task 6 (Monitor UI) | Yes | ~5-9h |
| 3 | Task 3 (Feature Extractor) + Task 5 (Gateway) | Yes | ~7-16h |
| 4 | Task 7 (Process Manager) + Task 8 (Replay Mode) | Yes | ~4h |
| 5 | Task 9 (Integration Tests) | -- | ~3h |
| **Total** | | | **~23h** |

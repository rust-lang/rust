# BCI-to-OpenClaw Integration Prototype -- Revised System Architecture (v2)

## 1. High-Level Overview

This system connects a Galea BCI headset (via BrainFlow SDK) to the OpenClaw AI agent platform, enabling the LLM to be aware of the user's real-time brain state. A Python process reads raw EEG data, extracts frequency-domain features, classifies brain state via heuristics, and serves it over a simple HTTP API. An OpenClaw TypeScript plugin fetches this state on every AI turn via the `before_prompt_build` lifecycle hook and injects a natural language summary into the LLM's system context. The architecture has 3 custom components; everything else (LLM routing, messaging, memory) is handled by OpenClaw natively.

## 2. Component Diagram

```
+------------------------------------------------------------------+
|           BCI Signal Processor + State Server                     |
|           (Python, single process)                                |
|                                                                   |
|  +---------------+     +-----------------+     +---------------+  |
|  | BrainFlow SDK |---->| DSP Pipeline    |---->| Classifier    |  |
|  | (synthetic or |     | (Welch's method,|     | (heuristic,   |  |
|  |  Galea board) |     |  band powers)   |     |  ML/sklearn,  |  |
|  +---------------+     +-----------------+     |  or EEGNet)   |  |
|    250 Hz EEG                |                 +---------------+  |
|                              v                        |           |
|                     +------------------+              v           |
|                     | Pause Detector   |     +-----------------+  |
|                     | (clench, drowsy, |     | State Manager   |  |
|                     |  headset off)    |     +-----------------+  |
|                     +------------------+         |    |    |      |
|                                                  v    v    v      |
|  +-------------------+ +------------------+ +-----------------+  |
|  | Session Recorder/ | | Webhook Manager  | | HTTP + WS Server|  |
|  | Replayer (JSONL)  | | (state-change    | | port 7680       |  |
|  +-------------------+ |  notifications)  | | CORS middleware |  |
|                        +------------------+ +-----------------+  |
+------------------------------------------------------------------+
          |                       |                     |
          |  HTTP POST callbacks  | HTTP GET /state     | WS /ws
          v                       v                     v
  +--------------+    +------------------------------+
  | External     |    | OpenClaw BCI Plugin (TS)     |
  | Webhook      |    |                              |
  | Consumers    |    |  api.on("before_prompt_build",|
  +--------------+    |    async () => {              |
                      |      fetch("/state");        |
                      |      return { prepend... };  |
                      |  })                          |
                      +------------------------------+
                                    |
                                    | prependSystemContext
                                    v
                      +------------------------------+
                      | OpenClaw Gateway (exists)    |
                      | port 18789                   |
                      |                              |
                      | - Routes to LLM             |
                      | - Loads BCI SKILL.md        |
                      | - Message flow, memory      |
                      +------------------------------+
```

## 3. Component Descriptions

### Component 1: BCI Signal Processor + State Server (Python)

**Single process combining data acquisition, DSP, and HTTP serving.**

- **BrainFlow integration:** Connects to synthetic board (prototype) or Galea hardware. Reads EEG at 250Hz.
- **DSP pipeline:** Buffers 1-second sliding windows (250 samples), computes band powers via Welch's method (delta/theta/alpha/beta/gamma), derives attention/relaxation/cognitive_load scores, estimates artifact probability, assesses signal quality.
- **Heuristic classifier:** Classifies brain state from band powers (high alpha = relaxed, high beta/theta ratio = focused, high theta = drowsy, etc.).
- **HTTP server:** FastAPI on port 7680. `GET /state` returns current `BCIState` JSON (conforms to `bci_state.schema.json`). `GET /health` returns server status.
- **Why one process:** KISS. The Signal Processor and State Server share the same state (latest brain features). Splitting them adds IPC complexity for zero benefit in a prototype.

### Component 2: OpenClaw BCI Plugin (TypeScript)

**A thin OpenClaw plugin that bridges BCI state into the agent's context.**

- Uses `definePluginEntry` / `register(api)` pattern.
- Registers a `before_prompt_build` lifecycle hook.
- On each hook invocation: fetches `GET http://127.0.0.1:7680/state`, extracts `natural_language_summary`, returns `{ prependSystemContext: summary }`.
- Handles errors gracefully: if state server is unreachable or data is stale, injects a warning instead of crashing.
- Configurable: state server URL can be overridden.

### Component 3: BCI SKILL.md

**A skill file teaching the OpenClaw agent how to interpret brain state data.**

- Loaded into the agent's context when relevant.
- Explains what each brain state means, what the scores represent, and how to adapt responses based on the user's cognitive state.
- Example: "When the user is drowsy, keep responses shorter and suggest taking a break."

## 4. Data Flow

```
Step  What                              Where
----  ----                              -----
 1    Raw EEG samples at 250 Hz         BrainFlow SDK -> numpy arrays
 2    Sliding window (1s, 250 samples)  DSP buffer in Signal Processor
 3    Band powers via Welch's method    scipy.signal.welch
 4    Derived scores (attention, etc.)  Simple ratios + normalization
 5    Brain state classification        Heuristic thresholds
 6    BCIState JSON + NL summary        State Server memory
 7    HTTP GET /state                   Plugin -> State Server
 8    prependSystemContext injection     Plugin -> OpenClaw Gateway
 9    LLM sees brain state in context   Gateway -> LLM provider
```

**Update frequency:**
- DSP: every 250ms (4 feature windows/second)
- State Server: latest state always available
- Plugin: fetches on each AI turn (user-driven, not continuous)

## 5. Technology Choices

| Choice | Justification |
|---|---|
| **Python** (Signal Processor + State Server) | BrainFlow's primary SDK; NumPy/SciPy for DSP; boring, well-known |
| **FastAPI** (HTTP server) | Async, fast, auto-generates OpenAPI docs, minimal boilerplate |
| **TypeScript** (OpenClaw plugin) | Required by OpenClaw plugin SDK; plugins loaded via jiti |
| **HTTP/JSON** (plugin -> state server) | Simplest transport for request/response. Plugin only reads on AI turns |
| **WebSocket** (real-time streaming) | For external dashboards or UIs that need continuous state updates |
| **BrainFlow** (BCI SDK) | Supports Galea + 200 other boards; synthetic mode for dev |
| **Heuristic classifier** (default) | No training data needed; transparent logic; always available |
| **scikit-learn** (ML classifier) | Optional Random Forest classifier trained via `train_model.py`; loaded from joblib model file at `BCI_MODEL_PATH`; falls back to heuristic if model missing |
| **braindecode + torch** (deep classifier) | Optional EEGNet classifier trained via `train_eegnet.py`; operates on raw EEG; install via `pip install .[deep]` |
| **CORS middleware** | FastAPI CORSMiddleware; origins configurable via `BCI_CORS_ORIGINS` env var (default `*`) |

**What we're NOT using:**
- ~~ZeroMQ~~ (HTTP is simpler for request/response)
- ~~Flask~~ (FastAPI is equally simple and async-native)
- ~~rich terminal UI~~ (the LLM is the consumer, not a human terminal)

## 6. Interface Contracts

### State Server API (Python)

| Endpoint | Method | Response Schema | Description |
|---|---|---|---|
| `/state` | GET | `state_server_api.schema.json#/definitions/state_response` | Current brain state |
| `/health` | GET | `state_server_api.schema.json#/definitions/health_response` | Server health |
| `/pause` | GET | `{ paused, reason, since_ms }` | Pause detection state (clench, drowsiness, headset removed) |
| `/webhooks` | POST | `{ id, registered }` | Register a webhook for state-change notifications |
| `/webhooks` | GET | `list[WebhookInfo]` | List all registered webhooks |
| `/webhooks/{webhook_id}` | DELETE | `{ deleted }` | Unregister a webhook by ID |
| `/ws` | WebSocket | Streams `BCIState` JSON frames | Real-time BCI state streaming; sends current state on connect |
| `/ws/info` | GET | `{ connections }` | WebSocket connection count for monitoring |

**Base URL:** `http://127.0.0.1:7680`

**CORS:** All endpoints are served behind `CORSMiddleware`. Allowed origins are configurable via the `BCI_CORS_ORIGINS` environment variable (comma-separated, default `*`).

### OpenClaw Plugin -> Gateway

| Hook | Returns | Description |
|---|---|---|
| `before_prompt_build` | `{ prependSystemContext: string }` | Brain state NL summary injected before every AI turn |

### BCI SKILL.md -> Agent

Loaded on-demand when agent determines BCI context is relevant. Provides interpretation instructions.

## 7. Error Handling Strategy

**Principle: Fail fast, fail loudly, degrade gracefully for the LLM.**

| Component | Failure | Response |
|---|---|---|
| Signal Processor | BrainFlow device disconnected | Log error, retry 3x with backoff, exit after 30s. State Server returns `available: false`. |
| Signal Processor | NaN/Inf in DSP | Replace with 0, set artifact_probability=1, log warning. |
| Signal Processor | No data for > 2s | Set signal_quality=0, staleness_ms increases, state becomes "unknown". |
| State Server | Startup before any BCI data | Return `{ available: false, error: "No BCI data received yet" }`. |
| Plugin | State Server unreachable | Inject: "BCI state unavailable (state server not responding)". Don't crash. |
| Plugin | State data stale (> 5s) | Inject warning: "BCI state is stale (Xs old), may be unreliable". |
| Plugin | State Server returns available=false | Inject: "BCI device not connected". |

## 8. File Structure

```
bci-integration/
├── README.md
├── .env.example
├── .gitignore
├── Makefile
├── docs/
│   ├── architecture.md          (this document)
│   ├── high-level-changes.md
│   └── task-breakdown.md
├── schemas/
│   ├── bci_stream.schema.json
│   ├── processed_features.schema.json
│   ├── bci_state.schema.json
│   └── state_server_api.schema.json
├── scripts/
│   └── demo.sh                  # Demo launcher script
├── signal-processor/            # Python: BrainFlow + DSP + HTTP server
│   ├── pyproject.toml
│   ├── models/                  # Trained model files (.gitkeep)
│   │   └── .gitkeep
│   ├── src/
│   │   ├── __init__.py
│   │   ├── __main__.py          # Entry point
│   │   ├── config.py            # All configuration (env vars with defaults)
│   │   ├── brainflow_reader.py  # BrainFlow connection + data acquisition
│   │   ├── dsp.py               # Pure DSP functions (band powers, scores)
│   │   ├── classifier.py        # Heuristic + ML (sklearn) classifiers
│   │   ├── deep_classifier.py   # EEGNet classifier (braindecode/torch)
│   │   ├── data_loader.py       # Training data loader utilities
│   │   ├── train_model.py       # Train sklearn Random Forest model
│   │   ├── train_eegnet.py      # Train EEGNet deep learning model
│   │   ├── models.py            # Pydantic models (BCIState, responses, etc.)
│   │   ├── state_manager.py     # Thread-safe current state storage
│   │   ├── pause_detector.py    # Pause detection (clench, drowsy, headset off)
│   │   ├── recorder.py          # Session recording to JSONL
│   │   ├── replayer.py          # Session replay from JSONL
│   │   ├── webhook_manager.py   # Webhook registration + state-change firing
│   │   ├── ws_manager.py        # WebSocket connection manager + broadcast
│   │   └── server.py            # FastAPI app (9 endpoints, CORS, lifespan)
│   └── tests/
│       ├── __init__.py
│       ├── fixtures/
│       │   └── sample_session.jsonl
│       ├── test_dsp.py
│       ├── test_classifier.py
│       ├── test_deep_classifier.py
│       ├── test_server.py
│       ├── test_pause_detector.py
│       ├── test_replay.py
│       ├── test_webhooks.py
│       └── test_websocket.py
├── openclaw-plugin/             # TypeScript: OpenClaw plugin
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   └── index.ts             # Plugin entry point with before_prompt_build
│   └── tests/
│       └── index.test.ts
└── skill/                       # OpenClaw skill for BCI interpretation
    └── SKILL.md
```

## 9. BCI SKILL.md Outline

```yaml
---
name: bci-brain-state
description: Interprets real-time brain-computer interface (BCI) data injected into context. Helps adapt responses based on the user's cognitive state.
version: 0.1.0
user-invocable: false
disable-model-invocation: false
---
```

**Skill instructions would cover:**
- What each brain state means (focused, relaxed, stressed, drowsy, meditative, active)
- What the scores represent (attention 0-1, relaxation 0-1, cognitive_load 0-1)
- How to adapt responses based on state:
  - **Focused**: User is engaged. Provide detailed, in-depth responses.
  - **Relaxed**: User is calm. Conversational tone is fine.
  - **Stressed**: User may be frustrated. Be concise, supportive, clear.
  - **Drowsy**: User is fatigued. Keep it short, suggest breaks.
  - **Unknown/low signal**: Don't rely on BCI data, respond normally.
- When to mention brain state vs. use it silently
- Signal quality caveats (low quality = unreliable data)

## 10. Scope

### Implemented
- Signal Processor with BrainFlow synthetic board (no hardware needed)
- DSP: Welch's method band powers, simple derived scores
- Heuristic classifier (threshold-based, always available as default/fallback)
- ML classifier: scikit-learn Random Forest trained via `train_model.py`, loaded from `BCI_MODEL_PATH`
- Deep classifier: EEGNet (braindecode + torch) trained via `train_eegnet.py`, loaded from `BCI_DEEP_MODEL_PATH`; install via `pip install .[deep]`
- FastAPI state server with 9 endpoints (GET /state, GET /health, GET /pause, POST /webhooks, GET /webhooks, DELETE /webhooks/{id}, WS /ws, GET /ws/info) and CORS middleware
- WebSocket streaming for real-time UI dashboards (`/ws` endpoint, broadcasts every state update)
- Webhook integration for event-driven state-change notifications (`/webhooks` CRUD endpoints, configurable cooldown and timeout)
- Pause detection (jaw clench, drowsiness, headset removed) via `pause_detector.py` and `GET /pause`
- Session recording to JSONL and replay from JSONL (`recorder.py`, `replayer.py`)
- OpenClaw plugin with before_prompt_build hook
- BCI SKILL.md for agent interpretation
- Schema validation at State Server boundary
- Basic structured logging

### Future Work (Out of Scope)
- Real Galea hardware support (just change board ID)
- EMG/EOG/EDA/PPG feature extraction (prototype = EEG only)
- Artifact rejection (ICA, ASR)
- Custom OpenClaw Node registration (for node.invoke pattern)
- Multi-user / multi-device
- Auth/TLS on state server
- Persistent storage / time-series DB

## 11. Revised Task Breakdown

| Task | Description | Depends On | Estimate |
|---|---|---|---|
| 1 | Project scaffold: Python package, TS package, schemas, config | -- | 1h |
| 2 | DSP module: band powers, scores, classifier (pure functions + tests) | 1 | 3h |
| 3 | BrainFlow reader: synthetic board, data acquisition loop | 1 | 2h |
| 4 | State Server: FastAPI, GET /state, GET /health, schema validation | 1, 2 | 2h |
| 5 | Integration: connect reader -> DSP -> state manager -> server | 2, 3, 4 | 2h |
| 6 | OpenClaw plugin: before_prompt_build, fetch /state, error handling | 1, 4 | 2h |
| 7 | BCI SKILL.md: interpretation instructions for the agent | -- | 1h |
| 8 | End-to-end test: synthetic board -> state server -> plugin mock | 5, 6 | 2h |
| **Total** | | | **~15h** |

Compared to the original 9-task / 23h plan, this is **8 tasks / ~15h** -- simpler architecture means less work.

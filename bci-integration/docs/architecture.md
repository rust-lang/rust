# BCI-to-OpenClaw Integration Prototype -- Revised System Architecture (v2)

## 1. High-Level Overview

This system connects a Galea BCI headset (via BrainFlow SDK) to the OpenClaw AI agent platform, enabling the LLM to be aware of the user's real-time brain state. A Python process reads raw EEG data, extracts frequency-domain features, classifies brain state via heuristics, and serves it over a simple HTTP API. An OpenClaw TypeScript plugin fetches this state on every AI turn via the `before_prompt_build` lifecycle hook and injects a natural language summary into the LLM's system context. The architecture has 3 custom components; everything else (LLM routing, messaging, memory) is handled by OpenClaw natively.

## 2. Component Diagram

```
+---------------------------------------------+
|           BCI Signal Processor + State Server |
|           (Python, single process)            |
|                                               |
|  +---------------+     +-----------------+    |
|  | BrainFlow SDK |---->| DSP Pipeline    |    |
|  | (synthetic or |     | (Welch's method,|    |
|  |  Galea board) |     |  band powers,   |    |
|  +---------------+     |  heuristic      |    |
|    250 Hz EEG          |  classifier)    |    |
|                        +-----------------+    |
|                              |                |
|                              v                |
|                        +-----------------+    |
|                        | HTTP Server     |    |
|                        | GET /state      |    |
|                        | GET /health     |    |
|                        | port 7680       |    |
|                        +-----------------+    |
+---------------------------------------------+
                               |
                               | HTTP GET /state
                               v
+---------------------------------------------+
|           OpenClaw BCI Plugin (TypeScript)     |
|                                               |
|  api.on("before_prompt_build", async () => {  |
|    const state = await fetch("/state");       |
|    return { prependSystemContext: summary };   |
|  })                                           |
+---------------------------------------------+
                               |
                               | prependSystemContext
                               v
+---------------------------------------------+
|           OpenClaw Gateway (already exists)    |
|           port 18789                          |
|                                               |
|  - Routes to LLM (Claude, GPT, Ollama, etc.) |
|  - Loads BCI SKILL.md for interpretation      |
|  - Normal message flow, memory, sessions      |
+---------------------------------------------+
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
| **HTTP/JSON** (plugin -> state server) | Simplest possible transport. No WebSocket, no ZMQ, no streaming needed since the plugin only reads on AI turns |
| **BrainFlow** (BCI SDK) | Supports Galea + 200 other boards; synthetic mode for dev |
| **Heuristic classifier** (brain state) | No training data needed; transparent logic; good enough for prototype |

**What we're NOT using:**
- ~~ZeroMQ~~ (HTTP is simpler for request/response)
- ~~Flask~~ (FastAPI is equally simple and async-native)
- ~~scikit-learn~~ (heuristic is simpler than ML for prototype)
- ~~rich terminal UI~~ (the LLM is the consumer, not a human terminal)

## 6. Interface Contracts

### State Server API (Python)

| Endpoint | Method | Response Schema | Description |
|---|---|---|---|
| `/state` | GET | `state_server_api.schema.json#/definitions/state_response` | Current brain state |
| `/health` | GET | `state_server_api.schema.json#/definitions/health_response` | Server health |

**Base URL:** `http://127.0.0.1:7680`

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
├── docs/
│   ├── architecture.md          (this document)
│   ├── high-level-changes.md
│   └── task-breakdown.md        (updated)
├── schemas/
│   ├── bci_stream.schema.json
│   ├── processed_features.schema.json
│   ├── bci_state.schema.json
│   └── state_server_api.schema.json
├── signal-processor/            # Python: BrainFlow + DSP + HTTP server
│   ├── pyproject.toml
│   ├── src/
│   │   ├── __init__.py
│   │   ├── __main__.py          # Entry point
│   │   ├── config.py            # Port, sample rate, window size constants
│   │   ├── brainflow_reader.py  # BrainFlow connection + data acquisition
│   │   ├── dsp.py               # Pure DSP functions (band powers, scores)
│   │   ├── classifier.py        # Heuristic brain state classifier
│   │   ├── state_manager.py     # Thread-safe current state storage
│   │   └── server.py            # FastAPI app (GET /state, GET /health)
│   └── tests/
│       ├── test_dsp.py
│       ├── test_classifier.py
│       └── test_server.py
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

### Prototype (In Scope)
- Signal Processor with BrainFlow synthetic board (no hardware needed)
- DSP: Welch's method band powers, simple derived scores
- Heuristic classifier (threshold-based, no ML)
- FastAPI state server (GET /state, GET /health)
- OpenClaw plugin with before_prompt_build hook
- BCI SKILL.md for agent interpretation
- Schema validation at State Server boundary
- Basic structured logging

### Future Work (Out of Scope)
- Real Galea hardware support (just change board ID)
- Trained ML classifier (replace heuristic)
- EMG/EOG/EDA/PPG feature extraction (prototype = EEG only)
- Artifact rejection (ICA, ASR)
- Session recording/replay
- WebSocket streaming (for real-time UI dashboards)
- Custom OpenClaw Node registration (for node.invoke pattern)
- Webhook integration (for event-driven state change alerts)
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

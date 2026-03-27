# BCI-to-OpenClaw Integration Prototype -- System Architecture

## 1. High-Level Overview

This system streams raw biosignal data from a Galea headset (via BrainFlow SDK), extracts neurological features in real time, and feeds them to the OpenClaw model integration layer for brain-state classification. The architecture is a simple linear pipeline of five single-purpose processes communicating over local ZeroMQ sockets with JSON payloads conforming to four defined schemas. The prototype targets a single user, single device, running entirely on one machine, with a target end-to-end latency under 200ms from sample capture to classified brain state.

## 2. Component Diagram

```
 +------------------+       +------------------+       +------------------+
 |                  |  ZMQ  |                  |  ZMQ  |                  |
 |  1. BCI Reader   |------>|  2. Feature      |------>|  3. OpenClaw     |
 |  (BrainFlow)     | PUB/  |     Extractor    | PUB/  |     Gateway      |
 |                  | SUB   |                  | SUB   |                  |
 +------------------+       +------------------+       +------------------+
   Galea hardware             BCIStreamPacket -->        ProcessedFeatures -->
   --> BCIStreamPacket        ProcessedFeatures          ModelInput/ModelOutput
                                                               |
                                                               | HTTP/JSON
                                                               v
                                                  +------------------+
                                                  |                  |
                                                  |  4. Model        |
                                                  |     Backend      |
                                                  |  (scikit-learn)  |
                                                  |                  |
                                                  +------------------+
                                                               |
                                                               |
                                                               v
                                                  +------------------+
                                                  |                  |
                                                  |  5. Monitor UI   |
                                                  |  (terminal/web)  |
                                                  |                  |
                                                  +------------------+

 Data flow:  Galea --> [1] --> [2] --> [3] --> [4]
                                               |
                                              [5] (subscribes to [3] output)
```

## 3. Component Descriptions

### Component 1: BCI Reader
- **Responsibility:** Acquire raw data from Galea hardware and emit normalized `BCIStreamPacket` JSON.
- **Single job:** Translate BrainFlow's board-specific array format into the `bci_stream.schema.json` contract.
- **Language:** Python (BrainFlow has best Python support and examples).
- **Behavior:** Connects to the Galea board via BrainFlow, reads samples at 250 Hz, packages each sample (or small batch) as a `BCIStreamPacket`, publishes on a ZMQ PUB socket.
- **Also handles:** Device discovery, connection retry, synthetic/playback mode for development without hardware.

### Component 2: Feature Extractor
- **Responsibility:** Buffer raw BCI packets into sliding windows and compute the features defined in `processed_features.schema.json`.
- **Single job:** Turn raw time-series data into frequency-domain features and derived scores.
- **Language:** Python (NumPy/SciPy FFT).
- **Behavior:** Subscribes to BCI Reader's ZMQ stream, maintains a 1-second sliding window (250 samples), computes band powers (delta/theta/alpha/beta/gamma) via Welch's method, derives attention, relaxation, and cognitive load scores, estimates artifact probability, assesses channel signal quality. Publishes `ProcessedBCIFeatures` on its own ZMQ PUB socket. Emits one feature vector per 250ms window step (75% overlap).

### Component 3: OpenClaw Gateway
- **Responsibility:** Bridge the BCI feature stream to the model backend.
- **Single job:** Protocol translation and request orchestration between the streaming BCI world and the request/response model world.
- **Language:** Python.
- **Behavior:** Subscribes to Feature Extractor's ZMQ stream. For each feature vector: constructs an `OpenClawModelInput` (with `intent: "classify_state"` for the prototype), sends it to the Model Backend via HTTP POST, receives the `OpenClawModelOutput`, and publishes the result on a ZMQ PUB socket for downstream consumers.

### Component 4: Model Backend
- **Responsibility:** Run brain-state classification inference.
- **Single job:** Accept a `ModelInput`, return a `ModelOutput`.
- **Language:** Python, Flask.
- **Behavior:** Exposes a single `POST /classify` endpoint. Takes the 5 band powers plus derived scores as feature vector, runs through a pre-trained scikit-learn classifier (or threshold-based heuristic for initial prototype), returns `brain_state` with `primary_state` and `confidence`.

### Component 5: Monitor UI
- **Responsibility:** Visualize system state for the developer/operator.
- **Single job:** Display what the system is doing so humans can verify it works.
- **Language:** Python (terminal-based with `rich` library).
- **Behavior:** Subscribes to the OpenClaw Gateway's output ZMQ socket. Displays current brain state, band power bars, signal quality, latency metrics.

## 4. Data Flow

```
Step  Source              Message                     Destination
----  ------              -------                     -----------
 1    Galea hardware      Raw BrainFlow arrays        BCI Reader
 2    BCI Reader          BCIStreamPacket (JSON)      Feature Extractor
                          via ZMQ PUB tcp://localhost:5555
 3    Feature Extractor   ProcessedBCIFeatures (JSON) OpenClaw Gateway
                          via ZMQ PUB tcp://localhost:5556
 4    OpenClaw Gateway    OpenClawModelInput (JSON)   Model Backend
                          via HTTP POST http://localhost:8080/classify
 5    Model Backend       OpenClawModelOutput (JSON)  OpenClaw Gateway
                          via HTTP response
 6    OpenClaw Gateway    OpenClawModelOutput (JSON)  Monitor UI
                          via ZMQ PUB tcp://localhost:5557
```

### Timing Budget

| Stage | Budget |
|---|---|
| BCI Reader (sample to publish) | < 5ms |
| Feature Extractor (window to features) | < 30ms |
| OpenClaw Gateway (overhead) | < 5ms |
| Model Backend (inference) | < 50ms |
| **Total end-to-end** | **< 100ms** (plus 250ms window step lag) |

## 5. Technology Choices

| Choice | Justification |
|---|---|
| **Python** (all components) | BrainFlow's primary SDK; NumPy/SciPy for DSP; scikit-learn for ML; one language = one team. |
| **ZeroMQ** (inter-process) | Boring, fast, zero-config PUB/SUB. No broker needed. |
| **JSON** over ZMQ | Human-readable, schema-validated, debuggable. Fine for 4 msg/sec at feature level. |
| **HTTP/JSON** (gateway to model) | Model backends universally speak HTTP. Easy to swap later. |
| **Flask** (model server) | Simplest Python HTTP framework. One file, no magic. |
| **scikit-learn** (classifier) | No GPU required. Fast inference (< 1ms). Easy to iterate. |
| **rich** (terminal UI) | Zero-dependency terminal dashboards. |
| **jsonschema** (validation) | Validate messages at component boundaries. |

## 6. Interface Contracts

| Interface | Transport | Schema | Rate |
|---|---|---|---|
| BCI Reader -> Feature Extractor | ZMQ PUB/SUB `tcp://localhost:5555` | `bci_stream.schema.json` | 250 msg/s |
| Feature Extractor -> Gateway | ZMQ PUB/SUB `tcp://localhost:5556` | `processed_features.schema.json` | 4 msg/s |
| Gateway -> Model Backend | HTTP POST `http://localhost:8080/classify` | `model_input.schema.json` / `model_output.schema.json` | 4 req/s |
| Gateway -> Monitor UI | ZMQ PUB/SUB `tcp://localhost:5557` | `model_output.schema.json` | 4 msg/s |

## 7. Error Handling Strategy

**Principle: Fail fast, fail loudly.**

| Component | Failure Mode | Response |
|---|---|---|
| BCI Reader | Device disconnected | Log error, retry with backoff. Exit after 30s. |
| BCI Reader | BrainFlow SDK error | Log traceback, exit. Let supervisor restart. |
| Feature Extractor | No data > 2s | Emit features with `signal_quality: 0`, `artifact_probability: 1`. |
| Feature Extractor | NaN/Inf in computation | Replace with zero, set `artifact_probability: 1`, log warning. |
| Gateway | Model timeout (> 100ms) | Return `status: "timeout"`, `primary_state: "unknown"`. Skip cycle. |
| Gateway | Model unreachable | Return `status: "error"`. Log. Continue trying. |
| Model Backend | Invalid input | HTTP 400 with `status: "error"` and message. |
| Model Backend | Inference crash | HTTP 500. Flask catches exception. Log traceback. |

**System-level:** Use a Procfile/shell script for process supervision. Structured logging to stderr. Schema validation at boundaries (toggle-able). Health check at `GET /health`.

## 8. Scope

### Prototype (In Scope)
- BCI Reader with BrainFlow synthetic board (no hardware needed)
- Feature Extractor with Welch's method band powers
- Derived scores (attention, relaxation, cognitive load) via simple ratios
- OpenClaw Gateway with `classify_state` intent only
- Model Backend with threshold-based heuristic classifier
- Terminal Monitor UI (state, confidence, signal quality)
- JSON schema validation at boundaries
- Shell script to start/stop all components
- Replay mode (record session to file, play back)

### Future Work (Out of Scope)
- `predict_action` and `adapt_model` intents
- Trained ML classifier (replace heuristic)
- Multi-user / multi-device
- Auth, TLS, security
- Persistent storage
- Cloud deployment
- Web-based UI
- EMG/EOG/EDA/PPG feature extraction (prototype = EEG only)
- Artifact rejection (ICA, ASR)
- Binary serialization / performance optimization

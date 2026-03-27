# BCI-OpenClaw Integration: Schema Realignment -- High-Level Changes

## Problem

The original schemas and architecture were designed around a hypothetical custom pipeline:
a ZeroMQ-based linear chain of Python processes culminating in a custom "OpenClaw Gateway"
and a Flask "Model Backend." This does not match how OpenClaw actually works.

OpenClaw is a real AI agent platform with its own Node.js Gateway (port 18789),
WebSocket + JSON transport, and a plugin system. It already handles LLM routing natively.
Building a parallel gateway and model backend is redundant and architecturally wrong.

## What OpenClaw Actually Provides

- **Gateway:** Node.js server on port 18789 (already exists, do not rebuild)
- **LLM routing:** Native model dispatch (do not build a Flask model backend)
- **Plugin system:** `definePluginEntry` / `register(api)` with lifecycle hooks
- **`before_prompt_build` hook:** Fires before every AI turn, allows injecting
  `prependSystemContext` -- this is the integration point for BCI state
- **SKILL.md:** Markdown files that provide domain instructions to the agent

## Revised Architecture (3 Components Instead of 5)

```
+-----------------+                       +-------------------+
|  BCI Signal     |  (internal or HTTP)   |  BCI State        |
|  Processor      | --------------------> |  Server           |
|  (Python)       |    /state             |  (Python/FastAPI) |
+-----------------+                       +-------------------+
  BrainFlow SDK                                   ^
  + DSP                                           | HTTP GET /state
                                                  |
                                            +-------------------+
                                            |  OpenClaw BCI     |
                                            |  Plugin (TS)      |
                                            |  before_prompt_   |
                                            |  build hook       |
                                            +-------------------+
                                                  |
                                                  | prependSystemContext
                                                  v
                                            +-------------------+
                                            |  OpenClaw Gateway |
                                            |  (already exists) |
                                            +-------------------+
```

### Component 1: BCI Signal Processor (Python)
- Uses BrainFlow SDK (synthetic board for prototype)
- Reads EEG samples at 250 Hz
- Computes band powers via Welch's method in sliding windows
- Derives attention, relaxation, cognitive load scores
- Writes current state to the BCI State Server (or is the same process)

### Component 2: BCI State Server (Python, e.g. FastAPI)
- Simple HTTP server exposing `GET /state` and `GET /health`
- Returns the latest processed BCI state as JSON
- Conforms to `bci_state.schema.json` and `state_server_api.schema.json`
- May be the same process as the Signal Processor for simplicity

### Component 3: OpenClaw BCI Plugin (TypeScript)
- Uses `definePluginEntry` and `register(api)` pattern
- On `before_prompt_build`, fetches `GET /state` from the BCI State Server
- Injects brain state as `prependSystemContext` so the LLM sees it
- A `BCI_SKILL.md` tells the agent how to interpret the injected data

## Schema Changes

| File | Action | Rationale |
|------|--------|-----------|
| `bci_stream.schema.json` | Updated | Fixed `$id` to `urn:bci-integration:bci_stream`; bumped to v0.2.0 |
| `processed_features.schema.json` | Updated | Fixed `$id` to `urn:bci-integration:processed_features`; bumped to v0.2.0 |
| `bci_state.schema.json` | **New** | LLM-readable brain state for `before_prompt_build` injection |
| `state_server_api.schema.json` | **New** | HTTP API contract for BCI State Server |
| `model_input.schema.json` | **Deleted** | Custom model backend removed; OpenClaw handles LLM routing |
| `model_output.schema.json` | **Deleted** | Brain state classification moved into Signal Processor |

## Why These Changes Matter

1. **No invented infrastructure.** Removes redundant gateway and model server.
2. **Plugin-native integration.** Uses `before_prompt_build` (cf. Claude-Mem precedent).
3. **Simpler data flow.** Raw BCI -> DSP -> brain state -> HTTP -> plugin -> LLM context.
4. **Correct `$id` namespacing.** Uses `urn:bci-integration:*` instead of non-existent domain.
5. **LLM-readable state.** New `natural_language_summary` field for direct context injection.

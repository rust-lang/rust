/**
 * OpenClaw BCI Plugin
 *
 * Fetches real-time brain state from the BCI State Server and injects
 * a natural language summary into the agent's system context via the
 * before_prompt_build lifecycle hook.
 *
 * Supports optional WebSocket streaming (BCI_USE_WEBSOCKET=true) for
 * lower-latency state updates with automatic HTTP fallback.
 */

import { definePluginEntry } from "openclaw";

const STALE_THRESHOLD_MS = 5000;
const FETCH_TIMEOUT_MS = 2000;
const WS_RECONNECT_DELAY_MS = 2000;

interface PauseEvent {
  pause_type: string; // "deliberate" | "automatic"
  trigger: string; // "jaw_clench" | "drowsiness" | "headset_removed"
  confidence: number;
  timestamp_unix_ms: number;
  recommended_action: string; // "pause" | "slow_down" | "stop"
}

interface ResumeEvent {
  timestamp_unix_ms: number;
  reason: string;
}

interface BCIState {
  timestamp_unix_ms: number;
  session_id: string;
  device_id: string;
  state: { primary: string; confidence: number };
  scores: { attention?: number; relaxation?: number; cognitive_load?: number };
  signal_quality: number;
  staleness_ms?: number;
  natural_language_summary: string;
  pause_event?: PauseEvent;
  resume_event?: ResumeEvent;
  [key: string]: unknown;
}

interface BCIStateResponse {
  available: boolean;
  timestamp_unix_ms: number;
  bci_state?: BCIState;
  error?: string;
}

function getBaseUrl(): string {
  return process.env.BCI_STATE_SERVER_URL ?? "http://127.0.0.1:7680";
}

function getWsUrl(): string {
  const base = getBaseUrl();
  return base.replace(/^http/, "ws") + "/ws";
}

function useWebSocket(): boolean {
  return process.env.BCI_USE_WEBSOCKET === "true";
}

async function fetchState(baseUrl: string): Promise<BCIStateResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(`${baseUrl}/state`, { signal: controller.signal });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    return (await res.json()) as BCIStateResponse;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Manages a persistent WebSocket connection to the BCI state server.
 * Caches the latest state and auto-reconnects on disconnect.
 */
class BCIWebSocketClient {
  private ws: WebSocket | null = null;
  private cachedState: BCIState | null = null;
  private pauseState: PauseEvent | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private url: string;
  private closed = false;

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    if (this.closed) return;
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const state = JSON.parse(
            typeof event.data === "string" ? event.data : String(event.data),
          ) as BCIState;
          this.cachedState = state;

          // Cache deliberate pause events so the hook can surface them
          if (state.pause_event?.pause_type === "deliberate") {
            this.pauseState = state.pause_event;
          }
          if (state.resume_event) {
            this.pauseState = null;
          }
        } catch {
          // Ignore malformed messages
        }
      };

      this.ws.onclose = () => {
        this.ws = null;
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        // onclose will fire after onerror, triggering reconnect
        try {
          this.ws?.close();
        } catch {
          // ignore
        }
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.closed || this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, WS_RECONNECT_DELAY_MS);
  }

  getState(): BCIState | null {
    return this.cachedState;
  }

  getPauseState(): PauseEvent | null {
    return this.pauseState;
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  close(): void {
    this.closed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
  }
}

function applyPauseResumePrefix(state: BCIState, summary: string): string {
  if (state.resume_event) {
    summary = `[RESUMED: User has resumed. Continue normally.] ${summary}`;
  }

  if (state.pause_event) {
    if (state.pause_event.pause_type === "deliberate") {
      summary = `[PAUSE: User deliberately paused via brain signal (jaw clench). Wait for resume before continuing.] ${summary}`;
    } else if (state.pause_event.trigger === "drowsiness") {
      summary = `[NOTICE: User appears drowsy. Keep responses very brief. Consider suggesting a break.] ${summary}`;
    } else if (state.pause_event.trigger === "headset_removed") {
      summary = `[NOTICE: BCI headset was removed. Ignore brain state data.] ${summary}`;
    }
  }

  return summary;
}

function buildContext(response: BCIStateResponse): string {
  if (!response.available) {
    return "BCI: device not connected";
  }

  const bci = response.bci_state;
  if (!bci) {
    return "BCI: device not connected";
  }

  let summary = bci.natural_language_summary;

  if (bci.staleness_ms != null && bci.staleness_ms > STALE_THRESHOLD_MS) {
    const staleSec = (bci.staleness_ms / 1000).toFixed(1);
    summary = `[WARNING: BCI state is stale (${staleSec}s old), may be unreliable] ${summary}`;
  }

  summary = applyPauseResumePrefix(bci, summary);

  return summary;
}

function buildContextFromState(state: BCIState): string {
  let summary = state.natural_language_summary;

  if (state.staleness_ms != null && state.staleness_ms > STALE_THRESHOLD_MS) {
    const staleSec = (state.staleness_ms / 1000).toFixed(1);
    summary = `[WARNING: BCI state is stale (${staleSec}s old), may be unreliable] ${summary}`;
  }

  summary = applyPauseResumePrefix(state, summary);

  return summary;
}

export default definePluginEntry({
  name: "bci",
  register(api) {
    const baseUrl = getBaseUrl();
    let wsClient: BCIWebSocketClient | null = null;

    // If WebSocket mode is enabled, start a persistent connection
    if (useWebSocket()) {
      wsClient = new BCIWebSocketClient(getWsUrl());
      wsClient.connect();
    }

    // Lifecycle hook: inject brain state summary before every AI turn
    api.on("before_prompt_build", async () => {
      // Try WebSocket cached state first
      if (wsClient !== null) {
        const cached = wsClient.getState();
        if (cached !== null) {
          return { prependSystemContext: buildContextFromState(cached) };
        }
        // WebSocket has no cached state yet -- fall through to HTTP
      }

      // HTTP polling (default, or fallback when WS has no data)
      try {
        const response = await fetchState(baseUrl);
        return { prependSystemContext: buildContext(response) };
      } catch {
        return { prependSystemContext: "BCI: state server unavailable" };
      }
    });

    // Tool: bci.status - returns full JSON for detailed inspection
    api.registerTool("bci.status", {
      description: "Get detailed BCI brain state data (full JSON from the state server)",
      parameters: {},
      async execute() {
        try {
          const response = await fetchState(baseUrl);
          return { result: JSON.stringify(response, null, 2) };
        } catch (err) {
          return {
            result: JSON.stringify({
              error: "State server unavailable",
              details: err instanceof Error ? err.message : String(err),
            }),
          };
        }
      },
    });
  },
});

// Re-export helpers for testing
export {
  fetchState,
  buildContext,
  buildContextFromState,
  getBaseUrl,
  getWsUrl,
  useWebSocket,
  BCIWebSocketClient,
  STALE_THRESHOLD_MS,
  FETCH_TIMEOUT_MS,
  WS_RECONNECT_DELAY_MS,
};
export type { BCIStateResponse, BCIState, PauseEvent, ResumeEvent };

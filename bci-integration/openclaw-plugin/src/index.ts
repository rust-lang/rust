/**
 * OpenClaw BCI Plugin
 *
 * Fetches real-time brain state from the BCI State Server and injects
 * a natural language summary into the agent's system context via the
 * before_prompt_build lifecycle hook.
 */

import { definePluginEntry } from "openclaw";

const STALE_THRESHOLD_MS = 5000;
const FETCH_TIMEOUT_MS = 2000;

interface BCIStateResponse {
  available: boolean;
  timestamp_unix_ms: number;
  bci_state?: {
    timestamp_unix_ms: number;
    session_id: string;
    device_id: string;
    state: { primary: string; confidence: number };
    scores: { attention?: number; relaxation?: number; cognitive_load?: number };
    signal_quality: number;
    staleness_ms?: number;
    natural_language_summary: string;
    [key: string]: unknown;
  };
  error?: string;
}

function getBaseUrl(): string {
  return process.env.BCI_STATE_SERVER_URL ?? "http://127.0.0.1:7680";
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

  return summary;
}

export default definePluginEntry({
  name: "bci",
  register(api) {
    const baseUrl = getBaseUrl();

    // Lifecycle hook: inject brain state summary before every AI turn
    api.on("before_prompt_build", async () => {
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
export { fetchState, buildContext, getBaseUrl, STALE_THRESHOLD_MS, FETCH_TIMEOUT_MS };
export type { BCIStateResponse };

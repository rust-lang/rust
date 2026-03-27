import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { buildContext, STALE_THRESHOLD_MS } from "../src/index.js";
import type { BCIStateResponse } from "../src/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeStateResponse(overrides: Partial<BCIStateResponse> = {}): BCIStateResponse {
  return {
    available: true,
    timestamp_unix_ms: Date.now(),
    bci_state: {
      timestamp_unix_ms: Date.now(),
      session_id: "test-session",
      device_id: "synthetic-0",
      state: { primary: "focused", confidence: 0.85 },
      scores: { attention: 0.8, relaxation: 0.2, cognitive_load: 0.6 },
      signal_quality: 0.9,
      staleness_ms: 100,
      natural_language_summary:
        "User brain state: FOCUSED (confidence: 0.85, attention: 0.80, relaxation: 0.20, signal quality: good)",
    },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// buildContext
// ---------------------------------------------------------------------------

describe("buildContext", () => {
  it("returns natural_language_summary when state is available and fresh", () => {
    const resp = makeStateResponse();
    const ctx = buildContext(resp);
    expect(ctx).toContain("FOCUSED");
    expect(ctx).not.toContain("WARNING");
  });

  it("prepends staleness warning when staleness_ms > threshold", () => {
    const resp = makeStateResponse({
      bci_state: {
        ...makeStateResponse().bci_state!,
        staleness_ms: STALE_THRESHOLD_MS + 1000,
      },
    });
    const ctx = buildContext(resp);
    expect(ctx).toContain("WARNING");
    expect(ctx).toContain("stale");
    expect(ctx).toContain("FOCUSED");
  });

  it("returns device-not-connected when available is false", () => {
    const resp = makeStateResponse({ available: false, bci_state: undefined });
    const ctx = buildContext(resp);
    expect(ctx).toBe("BCI: device not connected");
  });

  it("returns device-not-connected when bci_state is missing", () => {
    const resp: BCIStateResponse = {
      available: true,
      timestamp_unix_ms: Date.now(),
    };
    const ctx = buildContext(resp);
    expect(ctx).toBe("BCI: device not connected");
  });
});

// ---------------------------------------------------------------------------
// fetch integration (mock global fetch)
// ---------------------------------------------------------------------------

describe("before_prompt_build hook behavior", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("returns summary on successful fetch", async () => {
    const resp = makeStateResponse();
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(resp),
    });

    // Simulate what the hook does
    const { fetchState } = await import("../src/index.js");
    const data = await fetchState("http://127.0.0.1:7680");
    const ctx = buildContext(data);
    expect(ctx).toContain("FOCUSED");
  });

  it("throws on network failure (hook catches this)", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("ECONNREFUSED"));

    const { fetchState } = await import("../src/index.js");
    await expect(fetchState("http://127.0.0.1:7680")).rejects.toThrow("ECONNREFUSED");
  });

  it("throws on non-ok HTTP status", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: () => Promise.resolve({}),
    });

    const { fetchState } = await import("../src/index.js");
    await expect(fetchState("http://127.0.0.1:7680")).rejects.toThrow("HTTP 503");
  });
});

// ---------------------------------------------------------------------------
// Environment variable config
// ---------------------------------------------------------------------------

describe("getBaseUrl", () => {
  const originalEnv = process.env.BCI_STATE_SERVER_URL;

  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env.BCI_STATE_SERVER_URL;
    } else {
      process.env.BCI_STATE_SERVER_URL = originalEnv;
    }
  });

  it("returns default URL when env var is not set", async () => {
    delete process.env.BCI_STATE_SERVER_URL;
    const { getBaseUrl } = await import("../src/index.js");
    expect(getBaseUrl()).toBe("http://127.0.0.1:7680");
  });

  it("returns custom URL from env var", async () => {
    process.env.BCI_STATE_SERVER_URL = "http://10.0.0.5:9000";
    const { getBaseUrl } = await import("../src/index.js");
    expect(getBaseUrl()).toBe("http://10.0.0.5:9000");
  });
});

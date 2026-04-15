# Inbox/Port Convergence Strategy (Issue 46)

Status: Proposed
Decision type: Architecture direction + migration plan

## 1. Executive decision

Recommended direction:
Layered convergence (Strategy 3).

Interpretation:
- Keep Inbox and Port as distinct public abstractions.
- Converge implementation around a shared internal message-queue substrate and unified readiness model.

Why:
- Current tree already expresses two different semantic jobs:
  - Port/channel: endpoint connection and capability-gated transport.
  - Inbox: receiver-owned arrival queue for typed delivery and lifecycle notifications.
- Hard collapse into one user-facing abstraction would either lose endpoint semantics or overcomplicate inbox ownership paths.

Answer to "Are these the same thing?":
They share mechanics, not purpose. Treat them as sibling views over a shared core.

## 2. Internal model options (Phase 3.1)

### Option A: Inbox built on Port

Model:
`Inbox = wrapper over Port (single-consumer queue)`

Pros:
- Reuses existing pollable VFS integration and wait logic.
- Fewer kernel queue types exposed.

Cons:
- Inbox now inherits connection/peer semantics it does not naturally need.
- Pushes ownership-first lifecycle events through endpoint framing.
- Risks semantic leakage (EPIPE/HUP-style behavior in inbox paths).

Assessment:
Viable but conceptually backwards for ownership-first delivery.

### Option B: Port built on Inbox pair

Model:
`Port = (A->B inbox) + (B->A inbox)`

Pros:
- Clean message-direction decomposition.
- Makes structured message semantics explicit.

Cons:
- Does not naturally model stream ring behavior currently present in port.
- Requires rework of endpoint liveness and poll flags.
- Could regress channel performance if stream and message paths are forced together poorly.

Assessment:
Viable for pure message channels, weak fit for current mixed stream+message port behavior.

### Option C: Shared core primitive with two views

Model:
`KernelMessageQueue` core with `InboxView` and `PortView` adapters

Pros:
- Preserves semantic clarity of both abstractions.
- Enables code reuse for queueing, wakeups, backpressure, diagnostics.
- Supports gradual migration with minimal syscall churn.

Cons:
- Slightly more internal layering complexity.
- Requires disciplined boundary docs to avoid leaky abstractions.

Assessment:
Best fit for current code and future evolution.

## 3. Ownership and lifetime model (Phase 3.2)

Proposed unified rules:

1. Buffer ownership
- Messages are owned by the queue core once enqueue succeeds.
- Sender-side handles/caps follow current channel policy (duplicate semantics unless explicit move API is introduced).

2. Task/process exit
- Inbox-bound process queues may outlive sender; sender identity is metadata only.
- Port endpoints observe peer death via reader/writer refcounts and surface EPIPE/HUP semantics.

3. Handle close
- Closing last endpoint for a port transitions queue state to peer-closed/hup and eventually destroys when both sides gone.
- Closing/removing inbox registry entry marks closed and wakes waiters; object drops when last Arc is released.

4. Message durability
- Messages may outlive sender task.
- Messages do not outlive queue object destruction.

## 4. Compatibility matrix (Phase 4.1)

Note on naming:
The codebase currently uses `SYS_CHANNEL_*` names for port/channel operations. `SYS_PORT_*` in this issue maps conceptually to `SYS_CHANNEL_*`.

### Kernel syscall surface

| API | Status | Rationale |
|---|---|---|
| `SYS_CHANNEL_SEND` (`SYS_PORT_SEND` equivalent) | ✅ Compatible | Keep as transport-level send on PortView |
| `SYS_CHANNEL_RECV` (`SYS_PORT_RECV` equivalent) | ✅ Compatible | Keep as transport-level recv on PortView |
| `SYS_CHANNEL_WAIT` (`SYS_PORT_WAIT` equivalent) | ⚠ Needs adapter | Already deprecated; bridge through `SYS_FD_FROM_HANDLE` + `SYS_FS_POLL` |
| `SYS_CHANNEL_SEND_MSG` | ✅ Compatible | Natural PortView structured-message operation |
| `SYS_CHANNEL_RECV_MSG` | ✅ Compatible | Natural PortView structured-message operation |
| `SYS_MSG_SEND` (typed inbox delivery path) | ⚠ Needs adapter | Route through shared queue core once introduced |
| `SYS_MSG_BROADCAST` | ⚠ Needs adapter | Same as above; fanout stays higher-layer |
| `SYS_FS_POLL` | ✅ Compatible | Canonical readiness API for both once InboxView gets FD bridge |
| `SYS_FD_FROM_HANDLE` | ✅ Compatible | Existing bridge for PortView poll integration |

### Userspace impact

| Area | Status | Notes |
|---|---|---|
| Existing channel users (`channel_send/recv`) | ✅ Compatible | No API break required |
| Existing typed-message send/broadcast users | ✅ Compatible (behavioral) | Internal implementation can converge without changing call sites |
| Poll-based IPC loops | ⚠ Needs adapter for inbox | Add inbox-FD wrapper to make inbox pollable |
| Legacy `channel_wait` users | ⚠ Needs migration | Move to `fd_from_handle + fs_poll` |

## 5. Migration strategies (Phase 4.2)

### Strategy 1: Soft convergence (low risk)

Plan:
- Keep both abstractions and existing syscalls.
- Introduce adapters:
  - `port_to_inbox_view(...)` (internal first)
  - `inbox_to_pollable_fd(...)` (user-visible if needed)
- Migrate internals to shared queue core incrementally.

Pros:
- Minimal breakage.
- Easy rollback points.

Cons:
- Temporary duplication while adapters mature.

### Strategy 2: Hard convergence (high risk)

Plan:
- Deprecate either Inbox or Port public API.
- Rewrite call sites and syscall contracts to one abstraction.

Pros:
- Single conceptual model on paper.

Cons:
- High migration cost and regression risk.
- Does not align well with existing distinct semantics.

Recommendation:
Do not choose for current phase.

### Strategy 3: Layered model (recommended)

Plan:
- Define a shared `KernelMessageQueue` internal contract.
- Keep two explicit external views:
  - PortView = connection transport.
  - InboxView = ownership arrival queue.
- Unify readiness via VFS poll wrappers.

Pros:
- Matches current architectural reality.
- Gives clean boundary for future agents and docs.

Cons:
- Requires careful internal API design.

## 6. Actionable implementation roadmap

Phase A: Spec lock
1. Land semantics doc and this strategy doc.
2. Add an explicit glossary note: Port is connection-first, Inbox is ownership-first.

Phase B: Shared core extraction
1. Introduce internal queue trait/object used by both inbox and port message path.
2. Move shared backpressure, wakeup, queue metrics into the core.
3. Keep stream ring behavior in PortView-specific layer.

Phase C: Readiness unification
1. Add inbox VFS wrapper node (poll + waiter hooks).
2. Expose inbox FD acquisition path (syscall or path-open model).
3. Add tests for mixed poll sets: files + channels + inbox FDs.

Phase D: Adapter and deprecation cleanup
1. Document migration off deprecated `SYS_CHANNEL_WAIT`.
2. Keep stable channel syscalls; no forced user rewrite.
3. Optional: add explicit conversion helper APIs in userspace libraries.

## 7. Optional prototype scope (Phase 5)

Minimal feasibility prototype:
- Shared queue core with enqueue/dequeue/waiter/backpressure.
- One InboxView using that core.
- One PortView structured-message path using that core (stream path unchanged initially).
- Poll integration proof via inbox wrapper FD and `SYS_FS_POLL`.

Success criteria for prototype:
- Cross-communication sanity checks pass.
- Blocking behavior unchanged for existing channel users.
- Poll readiness behaves consistently for port and inbox wrappers.

## 8. Risks and mitigations

Risk: semantic drift between views.
Mitigation: document invariants in one shared contract; add assertion tests.

Risk: performance regressions in hot channel paths.
Mitigation: keep stream fast path separate initially; benchmark before merging paths.

Risk: user confusion from mixed terminology.
Mitigation: standardize docs and syscall comments around "channel" externally, "port" as kernel-private implementation term.

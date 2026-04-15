# Delivery Semantics Matrix — Thing-OS

**Status:** Architectural Reference  
**Scope:** Event vs Message delivery dimensions in the current Thing-OS tree  
**Issue:** #44  
**See also:** `docs/messaging/event-vs-message.md`

---

## 1. Core Dimension Comparison

Legend:
- ✓ — applies / supported
- ✗ — does not apply / not supported
- ~ — depends on class or configuration
- ⊕ — applies only via explicit adapter

| Dimension | Message | Event | Notes |
|-----------|---------|-------|-------|
| **Addressed to a specific receiver** | ✓ | ✗ | Messages name a destination; Events name a topic |
| **Sender specifies receiver** | ✓ | ✗ | Event publishers do not enumerate subscribers |
| **Consumptive delivery** | ✓ | ✗ | One receiver dequeues a message; observers read events non-destructively |
| **Multiple observers** | ✗ by default | ✓ | Broadcast Message delivers copies; Event fans out natively |
| **Implies work assignment** | ✓ | ✗ | Commands and RPC are Message-class |
| **FIFO ordering guaranteed** | ✓ | ~ | Edge Events may coalesce; State Events have no queue; Audit Events are ordered |
| **Backpressure on sender** | ✓ | ~ | Message queues reject when full; Events may coalesce or drop (class-dependent) |
| **Bounded queue** | ✓ | ~ | Inbox and channel have explicit capacity; Event subscriptions may be unbounded or use ring buffers |
| **Loss tolerated** | ✗ | ~ | Edge Events tolerate loss; Audit Events do not; State Events expose last-known-value |
| **Replayable** | ~ | ~ | Messages: only if persistence layer added; Audit Events: yes; Edge Events: no |
| **Coalescing allowed** | ✗ | ~ | Edge Events only; never for Audit Events; never for Messages |
| **Durable by default** | ✗ | ✗ | Both require explicit durability; Audit Events require it by class |
| **Late-subscriber sees history** | ✗ | ~ | Audit Events: yes; State Events: last-value; Edge Events: no |
| **Poll/readiness surface** | Wake Event (indirect) | ✓ | A readable inbox fires a Wake Event; event watches are directly pollable |
| **Blocking semantics** | ✓ | ✓ | Both can park a waiting task via `SYS_FS_POLL` or `SYS_WAIT_MANY` |
| **VFS thing exposure** | ~ | ✓ | Channels are VFS things; Inbox needs `InboxNode` wrapper (planned); watches are VFS things |
| **Capability passing** | ✓ | ✗ | `channel_send_msg` transfers things; Events do not carry capability handles |
| **Sender identity recorded** | ~ | ~ | Both support metadata; neither mandates it |
| **Authority model** | Capability (send-side thing) | Subscription registration | Sending a Message requires owning the endpoint; subscribing to an Event requires a registration API |

---

## 2. Delivery Class Breakdown

### 2.1 Message classes

| Class | Ordering | Loss tolerance | Backpressure | Replay | Typical primitive |
|-------|----------|---------------|-------------|--------|------------------|
| **Command Message** | FIFO | Intolerable | Hard reject on full | Not inherent | Channel (`SYS_CHANNEL_*`) |
| **Request/Reply Message** | FIFO, correlated | Intolerable | Hard reject on full | Not inherent | Channel + `abi::rpc::RpcHeader` |
| **Mailbox Message** | FIFO | Intolerable (acknowledged on enqueue) | Hard reject on full | Not inherent | Inbox (`SYS_MSG_SEND`) |

### 2.2 Event classes

| Class | Ordering | Loss tolerance | Coalescing | Replay | Typical primitive |
|-------|----------|---------------|-----------|--------|------------------|
| **Edge Event** | Best-effort | Tolerated if no subscriber | Allowed | Not replayed | Signal, IRQ readiness, input motion |
| **State Event** | Not applicable (last-value) | Low | Not applicable | Implicit via last-value read | VFS watch on a state file, `WaitKind::TaskExit` |
| **Audit Event** | Strict append | Intolerable | Not allowed | Required from cursor | Audit log VFS resource (planned) |
| **Wake Event** | Not applicable (level-triggered) | Not applicable | Level-triggering is its own coalescing | Not replayed | `POLLIN` / `POLLHUP` on a thing |

---

## 3. Primitive Mapping

### 3.1 Full classification table

| Primitive | Semantic class | Delivery class | Fanout | Consumptive | Pollable | Notes |
|-----------|---------------|---------------|--------|-------------|---------|-------|
| **Inbox** (`kernel::inbox`) | Message-native | Mailbox Message | ✗ (per-inbox copy) | ✓ | ~ (needs `InboxNode`) | Single-owner FIFO; `InboxNode` VFS wrapper planned |
| **Channel** (`SYS_CHANNEL_*`) | Message-native | Command / Request-Reply | ✗ | ✓ | ✓ via `SYS_FD_FROM_HANDLE` | Full poll integration today |
| **`SYS_MSG_SEND`** | Message-native | Mailbox Message | ✗ | ✓ | N/A (kernel delivery) | Directs to process inbox by pid |
| **`SYS_MSG_BROADCAST`** | Message-native (fanout delivery) | Mailbox Message × N | ✓ (inbox copy per member) | ✓ per recipient | N/A | Each recipient's copy is a Message; fanout is a delivery strategy, not an event model |
| **Pipe** (`SYS_PIPE`) | Message-native (byte-stream) | Command (stream) | ✗ | ✓ | ✓ | No message boundaries; streaming only |
| **Unix socket** (`SYS_SOCKET + AF_UNIX`) | Message-native (byte-stream) | Command (stream) | ✗ | ✓ | ✓ | Bidirectional byte stream; no multicast |
| **VFS RPC** (`SYS_FS_MOUNT` + channel) | Message-native | Request/Reply | ✗ | ✓ | ✓ (channel side) | Kernel serialises ops; provider owns reply |
| **VFS watch** | Event-native | Edge / State | ✓ | ✗ | ✓ | `SYS_FS_POLL`-ready; multiple watchers see same change |
| **`WaitKind::TaskExit`** in `SYS_WAIT_MANY` | Event-native | State Event | ✓ | ✗ | ✓ (level-triggered) | Once dead, status permanently observable |
| **Signal** (`kernel::signal`) | Event-native | Edge Event | ~ (targeted per-process) | ✗ | ✗ (async delivery) | Non-RT signals coalesce; RT signals queue but not yet implemented |
| **IRQ readiness** (`WaitKind::Irq`) | Event-native | Edge Event | ~ | ✗ | ✓ | Observation does not prevent other observers in same window |
| **`SYS_FS_POLL` / `SYS_WAIT_MANY` readiness** | Wake Event (neutral) | Wake Event | N/A | N/A | N/A | Readiness multiplexer; semantic payload is in the resource, not the poll call |
| **Futex** | Synchronisation primitive | Not Message or Event | ✗ | N/A | ✗ | In-process only; not a communication channel |

### 3.2 Ambiguous cases and resolutions

| Case | Issue | Resolution |
|------|-------|-----------|
| Channel used for event notification stream | Consumptive delivery breaks multi-observer intent | Wrap in a subscription service that fans out to per-subscriber channels, or switch to a watch-thing interface |
| `POLLIN` on an inbox/channel | Looks like an event; is actually Wake Event about Message availability | Correct behavior per Rule A; document explicitly; do not confuse the readiness signal with the Message payload |
| Group broadcast vs event fanout | `SYS_MSG_BROADCAST` fans out but delivers Message copies | Keep as Message with fanout-delivery-strategy tag; do not repurpose as Event bus |
| Subscription channel thing | A channel dedicated to event delivery (e.g., compositor surface events) | Classify as Event-class transport; use `KindId` to mark event records; do not mix with Command Messages on same channel |

---

## 4. Addressing and Sender-Intent Matrix

| Dimension | Command Message | Request/Reply Message | Mailbox Message | Edge Event | State Event | Audit Event | Wake Event |
|-----------|----------------|----------------------|----------------|-----------|------------|------------|-----------|
| **Receiver specified at send** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Sender intent** | Direct work | Exchange | Deliver and forget | Announce fact | Announce condition | Record fact | Signal readiness |
| **Reply expected** | ~ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Multiple receivers** | ✗ | ✗ | ✗ (one per queue) | ✓ | ✓ | ✓ | ✓ (per waiter) |
| **Delivery consumes** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Ordering critical** | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |

---

## 5. Queue Semantics Matrix

| Property | Inbox | Channel | Watch stream | Signal | Audit log |
|----------|-------|---------|-------------|--------|-----------|
| **Queue type** | FIFO bounded | FIFO bounded ring | Ring / last-value | Bitmask (non-RT) | Append-only log |
| **Capacity limit** | Yes — hard reject on full | Yes — hard reject on full | Ring overflow drops oldest (Edge) or retains last-value (State) | 1 bit per signal | Bounded by log policy |
| **Backpressure to sender** | ✓ | ✓ | ✗ (publisher does not block) | ✗ | ✗ |
| **Drain-before-close** | ✓ | ✓ | ✗ | N/A | ✓ |
| **Multi-consumer** | ✗ | ✗ | ✓ | Per-process bitmask (not shared) | ✓ (cursor per reader) |
| **Coalescing** | ✗ | ✗ | ✓ (Edge Events) | ✓ (inherent bitmask) | ✗ |

---

## 6. Readiness / Poll Behavior

| Resource | `POLLIN` means | `POLLOUT` means | `POLLHUP` means | Semantic class |
|----------|---------------|----------------|----------------|---------------|
| Inbox thing (future `InboxNode`) | Messages waiting to dequeue | Queue has capacity | Inbox closed | Wake Event about Message |
| Channel read thing | Messages waiting to dequeue | N/A | Writer closed | Wake Event about Message |
| Channel write thing | N/A | Queue has free space | Reader closed (`EPIPE`) | Wake Event about Message |
| Pipe read thing | Bytes available | N/A | Writer closed (EOF) | Wake Event about byte stream |
| Watch thing | Event records available | N/A | Watch invalidated | Event-native |
| Unix socket (connected) | Data from peer | Send buffer free | Peer closed | Wake Event about byte stream |
| Unix socket (listening) | Pending accept | N/A | N/A | Wake Event about connection |
| Regular VFS file | Always | Always | N/A | Always-ready |
| Task-exit thing (planned) | Task has exited | N/A | N/A | State Event (level-triggered) |

---

## 7. Subscription / Fanout Rules

| Rule | Messages | Events |
|------|---------|--------|
| Publisher enumerates receivers? | Yes (at send time) | No |
| Can one subscriber exclude others? | Yes (consumptive) | No (I-2) |
| Subscription is durable kernel object? | No — delivery is one-shot per send | Yes — subscription registration persists until deregistered |
| Late subscriber misses past? | Yes (queue drains) | Depends on event class (see §2.2) |
| Subscriber receives its own published fact? | Only if explicitly sent to self | By policy (may be filtered) |
| Unsubscribe removes buffered records? | N/A | Class-dependent; Edge Events: yes; Audit Events: cursor advances independently |

---

## 8. Compatibility: Shared vs Distinct vs Adaptable

| Dimension | Shared | Distinct | Adaptable with shim | Fundamentally incompatible |
|-----------|--------|---------|--------------------|-----------------------------|
| Envelope shape (`KindId` + payload bytes) | ✓ | | | |
| VFS thing / poll integration | ✓ | | | |
| Blocking / readiness model | ✓ | | | |
| Typed metadata header | ✓ | | | |
| Consumption semantics | | ✓ | | ✗ cannot share without distinguishing |
| Addressing model | | ✓ | | |
| Fanout vs point-to-point routing | | ✓ | ⊕ subscription service bridges | |
| Queue vs ring vs log storage | | ✓ | | |
| Backpressure model | | ✓ | | |
| Late-subscriber behavior | | ✓ | ⊕ last-value cache or audit log wraps edge source | |
| Coalescing | | ✓ | | ✗ Messages must not coalesce |
| Replay | | ~ | ⊕ explicit persistence layer | ✗ ephemeral Messages cannot be made replayable without opt-in |

---

## 9. Suitability by Use Case

| Use case | Best semantic class | Preferred primitive | Notes |
|----------|--------------------|--------------------|-------|
| Render-frame command to compositor | Command Message | Channel | One sender, one receiver, FIFO, backpressure |
| File changed under `/session/desktop/` | State Event | VFS watch | Multiple readers; last-value readable |
| Pointer motion from input driver | Edge Event | Watch thing or channel (subscription) | Loss tolerable; coalescing acceptable |
| `task_exited` lifecycle notification | State Event | `WaitKind::TaskExit` / watch handle | Level-triggered; multiple observers |
| RPC request to VFS provider | Request/Reply Message | VFS RPC channel | Strict correlated pair |
| Process group job-exit notification | Mailbox Message (fanout) | `SYS_MSG_BROADCAST` | Each recipient gets a Message copy; fanout is delivery strategy |
| Security audit log write | Audit Event | Audit log resource (VFS, append-only) | Durable, replayable, ordered |
| Input from stdin | Command Message (byte-stream) | Pipe | Sequential, no message boundaries |
| Service-ready announcement | State Event | VFS path under `/services/` + watch | Many potential consumers; last-value read on open |
| GUI window appeared / closed | Edge Event | Watch thing | Compositor publishes; many observers acceptable |
| Mutex acquire notification | Synchronisation (not IPC) | Futex | Not Message or Event; in-process only |
| Capability transfer between processes | Command Message + capability | `channel_send_msg` | Capability passing is Message-native |
| IRQ from hardware device | Edge Event | `WaitKind::Irq` | Driver-level; not user-visible directly |

---

## 10. See Also

- `docs/messaging/event-vs-message.md` — canonical definitions, interoperability rules, and implementation recommendations
- `docs/ipc/inbox_vs_port_semantics.md` — Inbox vs Port classification
- `docs/messaging/group-broadcast.md` — Group broadcast as Message delivery strategy
- `docs/concepts/ipc.md` — canonical IPC primitive overview and decision matrix
- `docs/concepts/readiness.md` — poll flag semantics per object class
- `docs/wait_many.md` — `SYS_WAIT_MANY` readiness kinds
- `docs/signals.md` — signal subsystem (Edge Event implementation)

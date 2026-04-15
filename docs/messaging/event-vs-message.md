# Event vs Message Semantics — Thing-OS Design Document

**Status:** Architectural Reference  
**Scope:** Kernel and userspace messaging semantics in the current Thing-OS tree  
**Issue:** #44  

> **Condensed slogan:** Messages ask. Events announce.

---

## 1. Canonical Definitions

### 1.1 Message

A **Message** is a directed transfer of intent or data from a sender to a
specific receiver or endpoint.

**Defining properties:**

- **Addressed** — a Message must name a destination (process, inbox, port
  endpoint, or typed route).  A message without a destination is not a message
  yet; it is a payload waiting to be wrapped.
- **Consumptive** — delivery is transactional.  The queue entry is removed when
  the receiver dequeues it.  Exactly one receiver observes each message under
  normal operation.
- **FIFO-ordered** — within a queue, ordering is meaningful.  Out-of-order
  delivery violates the delivery contract.
- **Intent-bearing** — a message implies work, acknowledgement, or coordinated
  state change on behalf of the sender.  The receiver is expected to act on
  it.
- **Backpressure-aware** — a full queue rejects or blocks the sender; the
  sender is responsible for the delivery commitment.

**Not properties of Messages:**

- Messages are not inherently durable (persistence is a delivery-class choice).
- Messages are not observable by third parties unless explicitly bridged.
- Messages are not broadcast by default.

**Examples:**

- `SYS_MSG_SEND(pid, kind, payload)` — deliver work to a process inbox.
- `channel_send_all(chan, b"render-frame")` — directed command over a channel.
- Request/reply RPC over a channel (both directions are Messages).
- Task A routes a job-exit notification to Task B's registered inbox.

---

### 1.2 Event

An **Event** is an observation that something happened, published so that
interested observers may react.

**Defining properties:**

- **Descriptive, not directive** — an event records a fact; it does not assign
  work to any specific receiver.
- **Fanout-oriented** — zero or more subscribers observe the same event.  The
  publisher does not need to enumerate receivers.
- **Potentially lossy by class** — depending on the event class (see §5),
  coalescing or dropping is acceptable when no subscriber is present or when
  the subscriber is too slow.
- **Unaddressed or loosely addressed** — a topic, path, or kind describes the
  event; the publisher does not specify individual receivers.
- **Observational** — observation does not consume the event for other
  observers.

**Not properties of Events:**

- Events are not inherently durable (durability is a delivery-class choice).
- Events are not consumptive by default.
- Events do not imply that the publisher is waiting for a response.

**Examples:**

- VFS watch fires when a file under `/session/desktop/` changes.
- Task-exited lifecycle notification broadcast to all registered watchers.
- Input device produces a pointer-motion record.
- Network link goes down; zero or more processes subscribed to link-state observe it.
- `bloom` receives a window-appeared event after the compositor mounts a new surface.

---

### 1.3 Supporting Definitions

| Term | Definition |
|------|-----------|
| **Subscription** | A registration expressing interest in a class of events by topic, kind, path, or filter predicate.  A subscription is not a queue; it describes routing intent. |
| **Delivery** | The act of transferring a message or event record to a specific receiver.  For a Message, delivery completes consumption.  For an Event, delivery updates an observer's view without removing the fact. |
| **Consumption** | Dequeuing a message such that no other receiver can obtain the same instance.  Consumption is a Message property, not an Event property. |
| **Observation** | Reading an event record from a subscription surface without consuming it for other observers.  Multiple observers may observe the same event. |
| **Replay** | Re-delivering earlier events to a new or catching-up subscriber.  Replay is meaningful only for durable event classes. |
| **Coalescing** | Merging multiple identical or superseding events into a single record before delivery (for example, multiple rapid file-change notifications collapsed into one). |
| **Fanout** | Delivering one logical event to all current subscribers.  Fanout is a routing behavior, not a storage behavior. |

---

## 2. When One Becomes the Other

### 2.1 Rule A — Messages may emit Events

Enqueueing a Message generates a readiness Event at the resource level.
When a message lands in an inbox or channel, the queue transitions to
`POLLIN`-ready.  This readiness signal is a **Wake Event** (§5.2); it is an
observation about Message availability, not the Message itself.

> The readiness event is not the message.  Poll observes that messages exist;
> read consumes them.

### 2.2 Rule B — Events are not Messages until explicitly bridged

A file-change watch notification is observable via a subscription surface
(a watch thing, a readable VFS node, or a `SYS_FS_POLL`-ready thing).  It
becomes a Message only when code explicitly bridges it: reading from the watch
FD, then sending a channel message to a specific service.

The bridge is an **adaptation point** (see §6); it is an explicit architectural
decision, not an automatic transformation.

### 2.3 Rule C — Directed subscription updates are still Events

A private subscription channel that receives task-exited notifications retains
Event semantics even though delivery is one-to-one.  The key invariant is
**observation rather than consumption**: if two subscribers hold handles to the
same event source, both should receive the notification.  Delivery may happen
to be single-subscriber in practice without changing the semantic class.

### 2.4 Rule D — Not all Events are durable

An input motion event that is dropped because no subscriber is registered is
acceptable.  A task-exit audit record that must be available to a late-joining
observer requires durability.  Event durability is a property of the **event
class** (§5), not of Events in general.

---

## 3. Primitive Classification

The following ThingOS primitives are classified against the model:

### 3.1 Message-native primitives

| Primitive | Classification | Rationale |
|-----------|---------------|-----------|
| **Inbox** (`InboxId`, `kernel::inbox`) | Message-native | Ownership-first bounded FIFO; addressed receiver; consumptive dequeue; backpressure on full. |
| **Channel** (`SYS_CHANNEL_*`, `Port`) | Message-native | Connection-oriented endpoint pair; consumptive reads; FIFO ordering; backpressure via bounded ring. |
| **`SYS_MSG_SEND`** | Message-native | Directed delivery to a process by `pid`; single consumer; consumptive. |
| **`SYS_MSG_BROADCAST`** | Message-native (fanout delivery) | Each recipient gets a private inbox copy — still consumptive per-recipient; broadcast is the delivery strategy, not an event fanout.  See §7 for the distinction. |
| **Unix domain sockets** | Message-native (byte-stream) | Stream delivery between two connected endpoints; no multicast; backpressure inherent. |
| **VFS RPC request/reply** | Message-native | Typed request → reply pairing across a channel; one provider, one consumer per operation. |

### 3.2 Event-native primitives

| Primitive | Classification | Rationale |
|-----------|---------------|-----------|
| **VFS watch / `SYS_FS_POLL` readiness** | Event-native | Observes resource state change; multiple watchers see the same change; non-consumptive. |
| **Signals** (`kernel::signal`) | Event-native (edge) | Asynchronous notification of a fact (SIGCHLD, SIGUSR1); no queue depth by default; coalescing allowed for non-realtime signals. |
| **Task-exit notification** | Event-native (state) | Once dead, a task's exit status is permanently observable; multiple observers may read it. |
| **`WaitKind::TaskExit`** in `SYS_WAIT_MANY` | Event-native (state) | Level-triggered: once task exits the readiness remains until cleared; any observer can see it. |
| **IRQ readiness** (`WaitKind::Irq`) | Event-native (edge) | Hardware interrupt occurred; observation does not preclude another observer from seeing the same event in the same delivery window. |

### 3.3 Overloaded or ambiguous primitives

| Primitive | Issue | Recommended resolution |
|-----------|-------|----------------------|
| **`SYS_WAIT_MANY` with `WaitKind::Fd`** | Routes FD readiness (Event) and Message queue availability (side-effect of Message arrival) through the same call. | This is correct.  `WAIT_MANY` observes readiness.  Readiness is always an Event; the underlying resource may hold Messages.  The overloading is intentional and benign (Rule A). |
| **Group broadcast** (`SYS_MSG_BROADCAST`) | Looks like fanout but uses Message inbox delivery. | Classify as Message-with-fanout-delivery-strategy.  Each recipient's inbox copy is a Message.  The broadcast act is a delivery mechanism, not an event fanout.  Add `delivery_kind = GroupBroadcast` metadata (already present) to distinguish at the application layer. |
| **Channel** used for event notification | A channel endpoint used to stream lifecycle events (e.g., compositor surface events) blurs Message and Event. | Introduce explicit typing via `KindId` or a convention field.  Consider wrapping in a subscription handle that names the event class so callers know observational vs consumptive semantics. |
| **`SYS_FS_POLL` `POLLIN` on a channel** | `POLLIN` signals "messages are waiting" which is a Wake Event about Message availability. | This is correct behavior under Rule A; document it explicitly in calling code to avoid confusion. |

---

## 4. Interoperability Rules

These rules are non-negotiable.  Violating them collapses the distinction.

| Rule | Statement | Consequence of violation |
|------|-----------|--------------------------|
| **I-1** | A Message must have an addressable destination at send time. | Without addressing, it is an event publish; routing logic breaks. |
| **I-2** | An Event must not be consumed (made unavailable) by a single observer. | If one reader destroys the event for others, it is a Message wearing an event name. |
| **I-3** | Readiness signals (poll/wake) are always Event-class, regardless of what resource they describe. | Conflating readiness with consumption introduces TOCTOU races. |
| **I-4** | Bridges between Event and Message are explicit adaptation points; there is no automatic promotion or demotion. | Implicit conversion hides semantic intent and makes auditing impossible. |
| **I-5** | Fanout delivery of inbox copies is Message semantics, not Event semantics. | Each recipient independently owns their inbox copy; one recipient failing does not roll back delivery to others. |
| **I-6** | A Message may carry metadata declaring that its arrival generates an Event (e.g., inbox-readable). | This is Rule A; it is allowed and expected.  The Event does not substitute for the Message. |

---

## 5. Delivery Classes

### 5.1 Message Classes

| Class | Description | Examples |
|-------|-------------|---------|
| **Command Message** | Directed work assignment; strong delivery expectation; sender may block or track acknowledgement. | `render-this-buffer`, `exec-this-binary`, kernel RPC request |
| **Request/Reply Message** | Correlated pair; sender expects a specific reply; correlation ID ties them. | VFS RPC op + response; `abi::rpc::RpcHeader`-framed IPC |
| **Mailbox Message** | Queued arrival for a named recipient; sender does not block for reply; delivery is fire-and-observe. | `SYS_MSG_SEND` to process inbox; job-exit delivery to observer inbox |

### 5.2 Event Classes

| Class | Description | Loss tolerance | Replayability | Examples |
|-------|-------------|---------------|--------------|---------|
| **Edge Event** | Something happened; no persistent state beyond the notification itself. | Tolerated if no subscriber; may coalesce. | Not replayed. | Input motion, IRQ fired, timer expired |
| **State Event** | Current condition; last-known-value semantics; late subscribers may read the current state. | Low; state should be readable on demand. | Implicit via last-value cache. | Network link state, battery level, window visibility |
| **Audit Event** | Append-only fact; must be durable; replayable for audit and late consumers. | Not tolerated. | Required. | Task lifecycle records, security events, filesystem mutations |
| **Wake Event** | Internal readiness signal tied to another resource; tells a waiter that a resource transitioned. | Only lost if waiter was not registered — never spuriously dropped. | Not replayed; level-triggered. | `POLLIN` on an inbox, `POLLHUP` on a channel |

---

## 6. Routing and Subscription Model

### 6.1 Message routing

Messages are routed by **destination address**, which may be:

- A process ID (`pid`) — resolved at send time against the live process table.
- An inbox ID (`InboxId`) — a kernel object handle.
- A channel endpoint — a thing number in the caller's thing table.
- A VFS path — resolved via the mount table to a service or device.

Routing authority: only the owner of a send-side endpoint (write thing, inbox
sender capability) may inject a message.  Capability passing via channels
(`channel_send_msg`) transfers routing authority explicitly.

Backpressure: full queues reject enqueues.  Callers must handle `EAGAIN` /
`SendError::Full`.

### 6.2 Event routing (subscription model)

Events are routed by **topic**, which may be:

- A **VFS path or path prefix** — watch a file or directory subtree.
- A **`KindId`** — a 16-byte typed kind identifier attached to the event
  record.
- A **lifecycle scope** — task, process group, session, or system.

Subscription resources:

- Watch handles (VFS watch) — returned by a watch-registration call; readable;
  expose event records as a stream.
- Subscription channel things — a channel endpoint written by the event source,
  read by the subscriber.
- `SYS_WAIT_MANY` entries with `WaitKind` matching an event source.

Fan-out method: the event source iterates registered subscriber handles and
delivers a copy to each.  Delivery to one subscriber does not affect others.

Late-subscriber behavior:

- **Edge Events** — not replayed; subscriber misses events before registration.
- **State Events** — last-known-value readable on demand from the event source
  (a VFS file, a device node, or a service endpoint).
- **Audit Events** — replay from position 0 or a cursor available via the
  audit log resource.
- **Wake Events** — level-triggered; a subscriber that registers after the
  transition immediately sees readiness.

Coalescing: permitted for Edge Events (for example, N rapid file-change events
before the subscriber drains may be collapsed into one).  Not permitted for
Audit Events.

Can one subscriber consume without depriving others? **No.**  This is rule I-2.
If a reader must consume an event for exclusivity, use a Message via an inbox
or channel.

Are subscriptions kernel objects, VFS nodes, or userland abstractions?

- Watch handles are kernel objects exposed as VFS things (level of
  `SYS_FS_POLL`-ready things).
- Subscription channels are kernel channel objects.
- Event multiplexer / filter services in userland may wrap these in higher-level
  subscription objects, but the kernel foundation is always a VFS thing +
  `SYS_FS_POLL`.

### 6.3 Routing model summary

| Dimension | Message | Event |
|-----------|---------|-------|
| Naming model | Destination endpoint (pid, InboxId, thing, path) | Topic (path, KindId, lifecycle scope) |
| Sender specifies receiver? | Yes | No |
| Routing authority | Capability (send-side thing ownership) | Subscription registration |
| Fanout by default? | No (broadcast is opt-in) | Yes |
| Late arrival | Missed; not replayed unless persistence layer added | Depends on event class |

---

## 7. Readiness and Poll Semantics

The cleanest invariant:

> **Poll concerns resource readiness. Poll does not carry semantic payload.**

Concretely:

- `SYS_FS_POLL` returns a readiness bitmask (`POLLIN`, `POLLOUT`, `POLLHUP`,
  `POLLERR`).  It does not return Message content or Event records.
- `POLLIN` on a channel or inbox thing means messages are available to be
  dequeued (a Wake Event about Message availability, per Rule A).
- `POLLIN` on a subscription watch thing means event records are available
  to be read.  Reading returns Event records; it does not consume the events
  for other subscribers.
- Watchable objects expose Events through their readable stream.
- Inboxes and channels expose readable Message queues.
- Poll/readiness integrates both — the distinction lives in the resource
  type behind the thing, not in the readiness flag semantics.

A caller that needs to wait on both Messages (inbox) and Events (watch) in one
loop may use `SYS_FS_POLL` or `SYS_WAIT_MANY` with mixed `WaitSpec` entries.
The poll loop itself is neutral; the semantics are in the resource.

---

## 8. Implementation Boundary Recommendations

### Canonical homes

| Primitive | Semantic home | Notes |
|-----------|--------------|-------|
| **Port / Channel** | Directed Messages | Connection-oriented, consumptive, backpressure-aware. |
| **Inbox** | Queued Messages | Ownership-first, single-consumer per inbox, bounded FIFO. |
| **Watch handle / subscription thing** | Events | Fanout, non-consumptive, class-dependent durability. |
| **`SYS_FS_POLL` / `SYS_WAIT_MANY`** | Readiness only | Neutral multiplexer; does not carry payload. |
| **Bridge/adapter** | Explicit Event↔Message conversion | Must be named, typed, and owned by a specific component. |
| **Signals** | Edge Events | Async, non-queued (non-RT), coalescing allowed. |
| **Audit log resource** | Audit Events | Durable, append-only, replayable; lives in VFS. |

### Do not

- Do not use a single queue abstraction for both Messages and Events without
  preserving the semantic distinction in type or behavior.
- Do not automatically promote an Event to a Message or demote a Message to an
  Event without an explicit adaptation point.
- Do not use `broadcast` delivery to substitute for event fanout design.
- Do not force all Events to be durable.
- Do not force all Messages to be replayable.
- Do not tie subscription semantics to process identity alone — resource and
  object semantics are more stable across task lifecycle changes.

### Recommended next steps (follow-on issues)

1. Introduce `InboxNode` as a pollable VFS wrapper for `Inbox` (poll
   integration gap identified in `docs/ipc/inbox_vs_port_semantics.md`).
2. Define a `WatchHandle` / subscription thing interface for Event sources so
   that every event source is a `SYS_FS_POLL`-ready VFS thing.
3. Introduce `KindId` convention for tagging event records so event-class can
   be read from the record header without inspecting payload.
4. Add explicit adaptation point API: `inbox_from_watch(watch_thing)` or
   equivalent bridge that converts an event stream into a message queue.
5. Classify all existing uses of `SYS_MSG_BROADCAST` vs event-fanout patterns
   and migrate pure notification paths to the Event model.
6. Design the Audit Event log resource (append-only, cursor-based, VFS-exposed).

---

## 9. Core Questions — Answered

| Question | Answer |
|----------|--------|
| Is an Event fundamentally a published fact while a Message is directed intent? | **Yes.** This is the foundational distinction. |
| Are Events allowed payload richness that resembles Messages? | **Yes.** An Event record may carry structured payload.  Rich payload does not make it a Message; the addressing and consumption model does. |
| Can Messages be transformed into Events after delivery/observation? | **Only via explicit bridge** (Rule B).  An observed message may generate an event record, but this requires an adaptation point, not automatic promotion. |
| Are Events consumable, replayable, broadcast, or merely observable? | **Depends on event class** (§5.2).  Edge Events: observable, may coalesce.  State Events: readable on demand.  Audit Events: durable and replayable.  Wake Events: level-triggered readiness. |
| Should one kernel primitive carry both, or should they sit on distinct rails? | **Distinct rails** (Channel/Inbox for Messages; watch things for Events) with readiness multiplexer (poll) as neutral glue. |
| What should poll/readiness mean for each? | Poll means **resource readiness only** — the resource behind the thing determines semantics.  See §7. |
| Do typed routes describe transport, semantics, or both? | Both.  A `KindId` on an envelope names the semantic class; the transport primitive (channel vs watch thing) describes the delivery contract. |
| What delivery guarantees belong to each class? | Messages: at-most-once per inbox slot, FIFO, backpressure on full.  Events: class-dependent — see §5.2 delivery class table. |

---

## 10. See Also

- `docs/messaging/delivery-semantics-matrix.md` — dimension-by-dimension comparison and primitive mapping
- `docs/ipc/inbox_vs_port_semantics.md` — Inbox vs Port analysis (Issue 46)
- `docs/messaging/group-broadcast.md` — Group broadcast as Message delivery strategy
- `docs/concepts/ipc.md` — canonical IPC primitive overview
- `docs/concepts/readiness.md` — poll/readiness flag semantics
- `docs/wait_many.md` — `SYS_WAIT_MANY` design and readiness kinds
- `docs/signals.md` — signal subsystem (edge event implementation)
- `abi/src/syscall.rs` — syscall numbers and ABI surface
- `kernel/src/inbox/mod.rs` — Inbox implementation
- `kernel/src/ipc/port.rs` — Port/channel implementation

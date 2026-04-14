# Group Broadcast via Typed Message Delivery — Design Note

**Status:** Prototype  
**Location:** `kernel/src/message/delivery.rs`

---

## 1. Goal

Validate that **group-level broadcast** can be expressed as a delivery strategy
layered on top of the same per-recipient typed-message enqueue path used for
direct delivery.  No new metaphysical channel or special-case group primitive is
introduced.

> Broadcast is a delivery strategy, not a separate metaphysical channel.

---

## 2. Group Definition

For this prototype a **Group** is the existing Unix process-group construct
(`pgid`) that lives in `Process.unix_compat.pgid`.  This is the narrowest
viable definition available without introducing a first-class kernel object.

| Concept        | Current source                         | Future direction      |
|----------------|----------------------------------------|-----------------------|
| Group identity | `Process.unix_compat.pgid` (u32)       | `thingos::group::Group` |
| Membership     | All processes with matching `pgid`     | `GroupKind`-aware membership |

---

## 3. Membership Semantics — Snapshot at Send Time

Membership is **snapshotted once at call time** before any per-recipient delivery
begins.

**Rationale:**

- Deterministic delivery count per broadcast.
- No lock is held across multiple enqueues; each enqueue acquires/releases its
  own recipient lock independently.
- Processes that join the group after the snapshot are not reached; this is the
  documented and tested behavior.

**Not chosen:** live iteration (would deliver to a moving target; harder to
reason about delivery count; non-deterministic for callers).

---

## 4. Message Shape

Each recipient receives a [`ProcessMessage`] carrying:

| Field                 | Value                                     |
|-----------------------|-------------------------------------------|
| `message.kind`        | Caller-supplied `KindId` (16 bytes)       |
| `message.payload`     | Caller-supplied opaque bytes              |
| `metadata.sender_tid` | TID of the sending task at enqueue time   |
| `metadata.sender_job` | PID of the sender's process, if known     |
| `metadata.target_group` | `pgid` of the target group             |
| `metadata.delivery_kind` | `MessageDeliveryKind::GroupBroadcast` |
| `metadata.broadcast_sequence` | 0-based index within the fanout  |

Direct delivery uses the same shape with `delivery_kind = Direct` and
`target_group = None`.

---

## 5. Failure Behavior

| Scenario                            | Behavior                                          |
|-------------------------------------|---------------------------------------------------|
| `pgid == 0`                         | `GroupBroadcastError::InvalidGroupId` → `EINVAL` |
| Empty group (no members)            | Success with zero deliveries; `targeted == 0`     |
| Recipient inbox full                | `DeliveryFailureReason::InboxFull`; fanout continues |
| Recipient exited before/during send | `DeliveryFailureReason::RecipientExited`; skipped |
| One failure does not block others   | Fanout continues; failure recorded in report      |

Partial failure is **explicit and observable** via `GroupBroadcastReport`:

```rust
pub struct GroupBroadcastReport {
    pub targeted:  usize,
    pub succeeded: usize,
    pub failed:    usize,
    pub failures:  Vec<RecipientFailure>,
}
```

---

## 6. API Surface

### Kernel-internal helper

```rust
// kernel/src/message/delivery.rs
pub fn deliver_typed_to_process(pid: u32, message: &Message)
    -> Result<(), DeliveryFailureReason>;

pub fn broadcast_typed_to_group_snapshot(pgid: u32, message: &Message)
    -> Result<GroupBroadcastReport, GroupBroadcastError>;
```

### Syscall boundary

| Syscall            | Number   | Args                                              |
|--------------------|----------|---------------------------------------------------|
| `SYS_MSG_SEND`     | `0x3030` | `(pid, kind_id_ptr, payload_ptr, payload_len)`    |
| `SYS_MSG_BROADCAST`| `0x3031` | `(pgid, kind_id_ptr, payload_ptr, payload_len)`   |

`SYS_MSG_BROADCAST` packs the outcome into the return value:
```
bits 31:16 — failures (saturated to 0xFFFF)
bits 15:0  — successes (saturated to 0xFFFF)
```

---

## 7. Shared Enqueue Boundary

Both direct and broadcast delivery converge on the same private helper:

```
deliver_typed_to_process         broadcast_typed_to_group_snapshot
         │                                │
         │                   snapshot_group_members_by_pgid
         │                                │
         └──────────── enqueue_to_process ─┘
                         (per-recipient lock + wake)
```

`enqueue_to_process` is the single canonical enqueue site for both paths.

---

## 8. Diagnostics

`broadcast_typed_to_group_snapshot` emits `kdebug!` log lines:

```
message::broadcast pgid=<N> targeted=<N> ok=<N> failed=<N>
message::broadcast pgid=<N> pid=<N> failure=<reason>    // per failure
```

---

## 9. Architectural Questions — Answers

1. **Can group broadcast be expressed as repeated typed delivery to current
   group members?** — Yes. `fanout_snapshot` iterates the snapshot and calls
   `enqueue_to_process` for each member.

2. **Where is the boundary?** — The boundary is `enqueue_to_process`.
   Membership resolution and metadata construction happen above it;
   queue manipulation and task wakeup happen inside it.

3. **Membership change during broadcast?** — Snapshot semantics: members
   captured at call time are the authoritative recipient set.

4. **Kernel vs service?** — The full fanout is currently inside the kernel
   for simplicity.  The `GroupBroadcastReport` return type preserves enough
   information for a future userspace coordination service to own this layer
   instead.

5. **Principled primitive?** — Yes.  Broadcast is implemented as
   `resolve_members + enqueue N ordinary messages`; there is no group magic.

---

## 10. Follow-On Tasks (Out of Scope)

- Unify direct / group delivery under `DeliveryTarget`
- Message IDs / receipts
- Sender exclusion rules
- First-class `thingos::group::Group` kernel object
- Userspace coordination service ownership of fanout
- Cross-job policy and priority scheduling

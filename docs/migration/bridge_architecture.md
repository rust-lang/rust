# Bridge Architecture: Old→New Semantic Translation Layer

> **Status**: Active migration infrastructure — Phase 9 baseline.
> Companion documents: `process_responsibility_map.md`, `boundary_audit.md`.

---

## Why bridges exist

Thing-OS is migrating incrementally from a Unix-derived Process/Thread model
toward a new ontology based on first-class canonical concepts: **Task**, **Job**,
**Group**, **Authority**, and **Place**.

The kernel still carries internal state in `Process` and `Thread<R>` structs —
that is the transitional backing.  Bridges are the **single conversion points**
that translate that internal state into the canonical generated types consumed
by procfs, IPC, syscall responses, and other public surfaces.

Without a disciplined bridge layer:

- Mapping logic gets duplicated across the codebase.
- Semantic drift goes undetected (old concepts leak into new code).
- Migration cannot be tested or reviewed in isolation.
- Each extraction phase becomes riskier and harder to validate.

Bridges contain the blast radius of changes.  When internal state moves, only
the bridge needs to change; all public-facing consumers remain stable.

---

## The rule: public meaning changes first, internals later

**New observation or IPC paths must use canonical types from the outset.**
Bridges are how kernel-internal state becomes canonical.

```text
kernel internal state
        │
        ▼
   bridge module  ←── single conversion point
        │
        ▼
canonical thingos type  ──► procfs / IPC / syscall response
```

New code must **never** read `Process` or `Thread` fields directly to produce
public-facing output.  Call the bridge.

---

## Bridge inventory

All currently active bridges are listed below.  Each bridge is the **canonical
public surface** for its domain; new code must go through the bridge, not read
`Process`/`Thread` fields directly.

| Bridge module                 | Domain      | Input (kernel-internal)                          | Output (canonical)                     | Phase |
|-------------------------------|-------------|--------------------------------------------------|----------------------------------------|-------|
| `kernel::task::bridge`        | Task        | `ThreadState`                                    | `thingos::task::Task`                  | pre-3 |
| `kernel::job::bridge`         | Job         | `ProcessSnapshot` + `ThreadState[]`              | `thingos::job::{Job,JobExit,JobWaitResult}` | 3 |
| `kernel::group::bridge`       | Group       | `ProcessSnapshot` (`pgid`/`sid`/`session_leader`)| `thingos::group::Group`                | 4     |
| `kernel::authority::bridge`   | Authority   | `ProcessSnapshot` (`name`, `exec_path`)          | `thingos::authority::Authority`        | 7     |
| `kernel::place::bridge`       | Place       | `ProcessSnapshot` (`cwd`, `namespace_label`)     | `thingos::place::Place`                | 8     |
| `kernel::message::bridge`     | Message     | raw `(KindId, Vec<u8>)`                          | `thingos::message::Message`            | -     |

### Planned (not yet introduced)

| Bridge module (planned)       | Domain      | Blocker                                         |
|-------------------------------|-------------|-------------------------------------------------|
| `kernel::space::bridge`       | Space       | `ProcessAddressSpace` subdivision exists; first-class `Space` object not yet introduced |
| `kernel::handle::bridge`      | Handle table| Handle-table concept not yet introduced         |
| `kernel::spawn::bridge`       | Spawn record| Spawn-record concept not yet introduced         |

---

## Conventions for bridge modules

Every bridge module follows these conventions so that contributors can find
and reason about translation logic quickly.

### File/module naming

```text
kernel/src/<concept>/bridge.rs
```

One module per canonical concept.  The file name is always `bridge.rs` inside
the concept's directory.

### Entry point naming

| Situation                                               | Preferred function name pattern           |
|---------------------------------------------------------|-------------------------------------------|
| Derive canonical type from `ProcessSnapshot`            | `<concept>_from_snapshot(snap)`           |
| Derive canonical type from raw kernel fields / slices   | `<concept>_from_<source>(…)`              |
| Convenience wrapper that calls the above                | `<concept>_from_<source>(…)` (same style) |
| Current-task entry point (Authority only today)         | `<concept>_for_current()`                 |

### Inputs and outputs

- **Inputs**: kernel-internal types only (`ThreadState`, `ProcessSnapshot`,
  `ProcessLifecycle`, raw slices, etc.).
- **Outputs**: schema-generated canonical types from the `thingos` crate only.
- No delivery, routing, or side-effect logic belongs inside a bridge.

### Tests

Each bridge module must contain a `#[cfg(test)]` block that:

1. Tests every public conversion function with at least the happy path and
   any documented edge cases (empty inputs, fallback behaviour, etc.).
2. Uses a local `make_snapshot` helper so tests are self-contained.
3. Does **not** assert on internal `Process`/`Thread` fields — only on the
   canonical output.

Tests live in the same file as the bridge code (inline `mod tests`), matching
the convention used by all existing bridge modules.

### Comments and PROVISIONAL markers

- Mark every mapping that relies on provisional kernel state with a
  `// PROVISIONAL:` comment explaining what changes when the field is properly
  extracted.
- Include a transitional mapping table in the module-level doc comment.
- Note which Migration Phase the bridge was introduced in.
- Describe the expected future removal path (e.g. "this bridge shrinks when
  `ProcessLifecycle` is promoted to a first-class `Job` object").

---

## Consuming bridges from procfs and syscall handlers

procfs paths and syscall handlers are the primary consumers of bridge output.

**Do this:**

```rust
// procfs handler for /proc/<pid>/authority
let authority = crate::authority::bridge::authority_from_snapshot(&snap);
let text = authority.as_text();
```

**Do not do this:**

```rust
// WRONG: reads Process fields directly — bypasses the bridge
let name = snap.name.clone();
let text = alloc::format!("name: {}\n", name);
```

The same rule applies to syscall response serialisation and any other
kernel-facing public surface.

---

## How long will bridges exist?

Bridges are **transitional infrastructure**.  Each bridge is expected to shrink
and eventually disappear as the underlying `Process`/`Thread` fields it reads
are migrated into first-class kernel objects.

The expected sequence per bridge:

1. Bridge reads from `Process`/`Thread` provisional backing (current state).
2. Backing fields move into a named subdivision (e.g. `ProcessLifecycle`).
3. Subdivision is promoted into a first-class kernel object (e.g. `Job`).
4. Bridge reads directly from the first-class object instead of `Process`.
5. All consumers remain unchanged — only the bridge internals change.
6. Once the first-class object is the sole source of truth, the bridge becomes
   a trivial forwarding shim and can eventually be inlined or removed.

No bridge should be removed while it is still the canonical public surface for
its domain.

---

## Guardrails

- **`kernel/src/boundary_contract.rs`** contains compile-time type checks that
  ensure every bridge function signature still returns the correct canonical
  type.  Changing a bridge return type away from the canonical type is a
  compile error.
- New boundary-facing bridge functions must be added to `boundary_contract.rs`
  before merging.
- Hand-written structs at boundary entry points that duplicate a canonical
  generated type are **not permitted**.

---

## Adding a new bridge

When a new canonical concept is introduced (e.g. `Space`):

1. Create `kernel/src/<concept>/bridge.rs`.
2. Declare `pub mod bridge;` in `kernel/src/<concept>/mod.rs`.
3. Add a module-level doc comment with the purpose, transitional mapping table,
   entry points, and future direction.
4. Implement at least one `<concept>_from_snapshot` function.
5. Add a `#[cfg(test)] mod tests` block with coverage for every public function.
6. Add the bridge function signature to `boundary_contract.rs`.
7. Update this document and `process_responsibility_map.md`.

---

## Related documents

- `docs/migration/process_responsibility_map.md` — canonical Process field
  inventory and extraction sequencing
- `docs/migration/boundary_audit.md` — audit of all kernel boundary surfaces
  against canonical generated types
- `docs/migration/authority_inventory.md` — detailed inventory of
  credential/permission fields
- `docs/migration/process_execution_context_inventory.md` — inventory of
  execution-context fields (cwd, namespace, tty, session)
- `docs/concepts/janix-guardrails.md` — architecture guardrails for all kernel
  changes

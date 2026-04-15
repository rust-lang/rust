# Space Object Design Document

**Phase:** Space Phase 1 (VM identity extraction)
**Status:** Implemented
**Tracking:** [Issue: Introduce Space as First-Class Address Space Object]

---

## What a Space Is

A **Space** is the canonical *virtual memory identity* in ThingOS.  It is the
explicit answer to the question:

> Where does memory live, and under what virtual mapping rules?

A Space owns or anchors:

| Owned                          | Status in Phase 1          |
|-------------------------------|----------------------------|
| Page-table root / arch VM context | Present (`aspace_raw`)  |
| VM mapping list (regions)     | Present (`mappings`)       |
| Stable identity (`SpaceId`)   | **New in Phase 1**         |
| COW/clone lineage metadata    | Future work                |
| Page-fault accounting counters| Future work                |

A Space does **not** imply:

| Concern                        | Canonical object            |
|-------------------------------|-----------------------------|
| Scheduler identity            | `Task` / `Thread`           |
| PID / TaskId                  | `Task` / `Thread`           |
| Lifecycle / exit tracking     | `Job`                       |
| Credentials / capabilities    | `Authority`                 |
| cwd / namespace / root        | `Place`                     |
| Session / process group       | `Group`                     |
| Open file descriptors         | Future resource authority   |
| Signals                       | Future authority concern    |

---

## Invariants

1. `Space.id` (`SpaceId`) is non-zero for every user-created `Space`.
   `SpaceId(0)` is reserved as `SpaceId::NONE` ("no space" / kernel thread).

2. `Space.mappings` is the authoritative list of virtual regions for this
   address space.  All `Thread`s running in this `Space` share the same
   `Arc<Mutex<MappingList>>` so mutations are immediately visible.

3. A `Space` is created exactly once and never recycled.  When all `Arc<Space>`
   references drop, the object is destroyed and the `SpaceId` is never reused
   within a boot session.

4. `Space.aspace_raw` holds the architecture-specific page-table token.  It is
   written only during spawn/exec (single writer) and read by the scheduler's
   fast path.

---

## Design Questions Answered

### Is Space the sole owner of a page-table root?

**Yes, in the intended model.** In Phase 1, `aspace_raw` is stored in both
`ProcessAddressSpace.aspace_raw` and `Space.aspace_raw`.  In future phases,
`ProcessAddressSpace` is removed and `Space` becomes the sole owner.

### Are mappings always attached to exactly one Space?

**Yes, for the canonical model.** In Phase 1 the `Arc<Mutex<MappingList>>` is
shared between `ProcessAddressSpace.mappings`, `Space.mappings`, and each
`Thread.mappings` (fast-path cache).  That shared arc represents one logical
address space.  Sharing the same arc across multiple `Space` instances would be
wrong; each `Space` should own a distinct `MappingList`.

### Can two tasks run in the same Space?

**Yes, this is a design goal.** In Phase 1, threads sharing one `Process` share
one `Space` via the same `Arc<Mutex<MappingList>>`.  In Phase 3, multiple `Task`
objects will hold `Arc<Space>` explicitly, enabling thread groups without the
`Process` wrapper.

### Can a Space survive task exit temporarily?

**Yes.** Because `Space` is reference-counted via `Arc<Space>`, it persists as
long as at least one holder exists.  A task may exit while another task in the
same thread group still holds the same `Space`.  The `Space` is destroyed when
its last `Arc` reference is dropped.

### Is exec modeled as Space replacement, Space mutation, or new Space creation?

**Phase 1 (current):** exec mutates the existing `Space` in place (replaces
`mappings` and `aspace_raw`).

**Intended direction (Phase 3):** exec creates a *new* `Space` and atomically
replaces the task's `Arc<Space>` reference, leaving the old space alive for any
other task that might reference it.  This enables cleaner fork-on-exec semantics.

### Is Space purely VM, or does it also own user heap/program-image metadata?

**Purely VM in Phase 1.** The `Space` owns the page-table root and mapping list.
Program image metadata (entry point, auxiliary vector, etc.) remains in
`ProcessUnixCompat` as legacy compatibility state and will be migrated to a
separate concept in a future phase.

### Must kernel userspace access always go through current Space?

**Yes.** All user-memory access is mediated through the current task's address
space (the `Space` loaded by the scheduler on context switch).  There is no
mechanism to borrow a reference to another task's `Space` for direct memory
access.  Future work may add a controlled "borrow" API for IPC zero-copy paths,
but that is explicitly out of scope for Phase 1.

### What is the Space ↔ memfd-backed shared mapping relationship?

**Future work.** Phase 1 does not introduce memfd.  When memfd is implemented,
a named mapping in one `Space` would reference the same physical pages as a
named mapping in another `Space` via a shared `Arc<MemfdBacking>`.  The
`Space` remains the unique virtual owner; sharing is expressed at the physical
backing level, not by sharing `Space` objects.

### Are Spaces nested/derived/snapshotted?

**Future work (not in Phase 1).**  COW-derived Spaces (fork-like semantics) are
out of scope.  Phase 1 establishes the object identity foundation that Phase 3
will use to implement clean derivation semantics.

### Does the scheduler need more than activate-this-Space-before-run?

**No, in the minimal model.** The scheduler only needs to load the page-table
root (`aspace_raw`) and update the per-CPU mapping cache when switching to a
task.  Both are available without locking through `Space.aspace_raw` and
`Space.mappings` respectively.

### Is Unix-style Process eventually a composite?

**Yes.** The long-term direction is that a Unix-compatible `Process` becomes a
composite of:

- one or more `Task`s (schedulable units)
- one `Space` (address space / VM domain)
- `Authority` (credentials)
- handle/fd table (open resources)
- `Place` (cwd, namespace)
- `Group` (session/signal routing)

None of these are fused; they reference each other.  `Process` becomes a
compatibility shim over this richer structure.

---

## Lifecycle and Ownership Rules

### Space Creation

A `Space` is created by `ProcessAddressSpace::empty()` or
`ProcessAddressSpace::from_parts(...)` at task/process spawn time.
`alloc_space_id()` assigns a unique monotonic `SpaceId`.

### Space Sharing

Multiple `Task`s may share one `Space` by holding clones of `Arc<Space>`.  In
Phase 1, all threads in a `Process` share the same `Space` via the shared
`mappings` `Arc`.

### Space Replacement (exec-like)

In Phase 1, exec mutates the `Space` in place: it clears `mappings` and updates
`aspace_raw`.  The `SpaceId` remains stable across exec in Phase 1 because the
same `Space` object is reused.

In Phase 3, exec will atomically replace the task's `Arc<Space>` with a freshly
created `Space`.  The old `Space` (with the old `SpaceId`) will remain alive
until all references drop.

### Last-Reference Teardown

`Space` is a plain Rust struct allocated behind `Arc`.  Teardown is ordinary
reference-counting: the last `drop` of the `Arc` runs the `Drop` impl (which
today is the default: nothing).  Future phases will add explicit teardown hooks
for architecture VM cleanup (page-table deallocation, TLB shootdown).

### Core Question: What Destroys a Space?

**Ordinary object lifetime (reference counting), not process mythology.**

---

## Migration Stance for fork/clone/exec

| Operation          | Phase 1 behavior                                    | Target behavior (Phase 3)                         |
|--------------------|----------------------------------------------------|----------------------------------------------------|
| `spawn`            | New `Space` created inside `ProcessAddressSpace`   | New `Space` created; task holds `Arc<Space>`       |
| exec               | Existing `Space` mutated in place                  | New `Space` created; task's `Arc<Space>` replaced  |
| fork-like          | Not supported yet                                  | New task with COW-derived `Space`                  |
| thread-like        | New `Thread` shares parent's `mappings` Arc        | New `Task` holds clone of parent's `Arc<Space>`    |

---

## Implementation Phases

### Phase 1 (implemented): Make Space explicit internally

- ✅ Introduce `kernel::space::Space` wrapping `mappings` and `aspace_raw`
- ✅ Assign stable `SpaceId` at construction time via `alloc_space_id()`
- ✅ Embed `Space` inside `ProcessAddressSpace.space_obj`
- ✅ Populate `ProcessSnapshot` with `space_id`, `space_mapping_count`,
  `space_sharing_count`
- ✅ Introduce `kernel::space::bridge` as the canonical public surface
- ✅ Add `thingos::space::Space` public canonical type
- ✅ Tests covering creation, sharing, Arc mutation visibility, and bridge

### Phase 2: Decouple APIs

- Refactor VM syscalls to call `Space`-backed helpers rather than `Process`
  methods directly
- Add `space_from_arc` call sites in context-switch and fault-handling paths
- Reduce direct `Process.space.mappings` manipulation in favor of
  `Process.space.space_obj.mappings`

### Phase 3: Enable sharing/replacement

- Replace `Process.space: ProcessAddressSpace` with `Process.space: Arc<Space>`
- Allow multiple `Task`s to hold `Arc<Space>` explicitly
- Implement exec as `Arc<Space>` replacement rather than in-place mutation
- Define COW-derived Space semantics for fork-like operations

### Phase 4: Observability and future ABI hooks

- Expose `Space` via a handle / capability for controlled cross-process mapping
- Add procfs visibility (`/proc/<pid>/space`)
- Prepare for future memfd / shared-mapping ABI anchored in `Space`

---

## Non-Goals / Guardrails

- **Do not** rename everything to `Space` while keeping `Process`-shaped
  ownership inside it.
- **Do not** fuse `Space` with credentials, job control, session, or environment
  state.
- **Do not** expose half-baked public handle ABI for `Space` before the object
  model is stable.
- **Do not** let Unix process-compatibility requirements dictate the internal
  architecture of `Space`.

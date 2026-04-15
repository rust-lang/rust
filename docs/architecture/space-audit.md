# Space Audit — Process / Task VM Coupling Inventory

**Phase:** Space Phase 1 (VM identity extraction)
**Status:** Initial audit
**Related:** `docs/architecture/space-object.md`

This document inventories all places in the kernel where address-space semantics
are implicitly attached to `Process` / `Task`, categorizes them by migration
urgency, and records the recommended disposition for each.

---

## Summary

| Category                            | Count | Action            |
|-------------------------------------|-------|-------------------|
| Must move (or adapter) now          | 3     | Phase 1 adapters added |
| Adapter layer acceptable            | 12    | Phase 2 target    |
| Defer until later ABI cleanup       | 8     | Phase 3+          |

---

## A. Must Move Now (or adapter required)

These are the coupling points that block a coherent Space object model.
Phase 1 has introduced adapter layers (`ProcessAddressSpace.space_obj`,
`ProcessSnapshot.space_*` fields, `kernel::space::bridge`) for all of them.

### A1. `ProcessAddressSpace` embedded in `Process`

**File:** `kernel/src/task/mod.rs`
**Field:** `Process.space: ProcessAddressSpace`
**Issue:** Address-space ownership is expressed as a nested struct inside
`Process` rather than as a reference to a first-class object.  There is no
stable `SpaceId` and no way to share a space across processes.

**Phase 1 adapter:** `ProcessAddressSpace.space_obj: Arc<Space>` is now
embedded inside `ProcessAddressSpace`.  The `space_obj` carries a stable
`SpaceId` and shares the same `Arc<Mutex<MappingList>>` as
`process.space.mappings`.

**Phase 3 target:** Replace `Process.space: ProcessAddressSpace` with
`Process.space: Arc<Space>`.  Remove `ProcessAddressSpace` entirely.

---

### A2. `ProcessSnapshot` has no address-space identity

**File:** `kernel/src/sched/hooks.rs`
**Type:** `ProcessSnapshot`
**Issue:** Snapshot used by procfs and bridge modules had no VM identity fields,
making it impossible to build a canonical `Space` from a snapshot.

**Phase 1 fix:** Added `space_id: SpaceId`, `space_mapping_count: u32`,
`space_sharing_count: u32` to `ProcessSnapshot`.  These are populated in
`list_processes` from `process.space.space_obj`.

---

### A3. No canonical `Space` type at the public boundary

**File:** `thingos/src/` (crate)
**Issue:** The `thingos` canonical-type crate had no `Space` type, so address-
space state could not be represented at system boundaries.

**Phase 1 fix:** Added `thingos::space::Space`, `thingos::space::SpaceId`,
and `kernel::space::bridge` as the single conversion point.

---

## B. Adapter Layer Acceptable (Phase 2 targets)

These coupling points do not block the Space object model but should be
migrated in Phase 2 as part of decoupling VM syscall APIs from `Process`.

### B1. `Thread.mappings` fast-path cache

**File:** `kernel/src/task/mod.rs`
**Field:** `Thread.mappings: Arc<Mutex<MappingList>>`
**Description:** Each `Thread` caches a clone of `Process.space.mappings` for
zero-lock scheduler access.  This is a deliberate performance optimization and
correctly shares the underlying `MappingList`.

**Coupling:** Thread holds a raw `Arc<Mutex<MappingList>>` rather than an
`Arc<Space>`.  The fast path bypasses `Space` identity entirely.

**Disposition:** Adapter acceptable.  The fast-path cache should eventually
be changed to `Thread.space: Arc<Space>` so context switch can obtain `SpaceId`
without a lock.  This enables future debug/accounting code to identify which
`Space` is active without reading `process_info`.

---

### B2. Context-switch / scheduler loading `aspace_raw` from `Thread.aspace`

**File:** `kernel/src/sched/` (architecture-specific context switch)
**Description:** On context switch, the scheduler loads the page-table root
from `Thread.aspace` (an architecture-typed token) rather than from an
explicit `Space` reference.

**Coupling:** The scheduler never reads `Space.aspace_raw` directly; it reads
the typed architecture token stored per-thread.

**Disposition:** Adapter acceptable for now.  Phase 2 can add a
`Space::activate()` hook that the scheduler calls on switch, abstracting the
architecture load behind `Space`'s interface.

---

### B3. `vm::add_user_mapping` targets the current process

**File:** `kernel/src/sched/vm.rs` (and related)
**Description:** VM-map syscall helpers (`add_user_mapping`, `remove_user_mappings`,
`check_user_mapping`, `get_user_mapping_at`, `protect_user_range`) all operate
on the current process's `mappings` via the scheduler hook.

**Coupling:** The "current process" identity is implicit; no `SpaceId` or
`Arc<Space>` is involved.  Two threads in the same process map into the same
`MappingList` by accident of sharing the same `Process`.

**Disposition:** Phase 2 should refactor these to accept an `Arc<Space>` (or
derive it from the current task's `process.space.space_obj`) so the target
space is explicit in every VM operation.

---

### B4. Page-fault handler in `kernel_handle_page_fault`

**File:** `kernel/src/lib.rs`
**Description:** The page fault handler calls `sched::handle_user_stack_fault_current`
which implicitly resolves the faulting address against the current task's
address space.  There is no explicit `Space` reference in the fault path.

**Coupling:** The fault path never names a `Space`; it operates through implicit
current-task identity.

**Disposition:** Phase 2 should introduce a `Space::handle_fault(addr)` entry
point that the fault handler calls after obtaining `Arc<Space>` from the
scheduler.

---

### B5. ELF loading / exec path mutates `Process.space` in place

**File:** `kernel/src/sched/spawn.rs`, `kernel/src/task/exec.rs`
**Description:** ELF loading populates `Process.space.mappings` and
`Process.space.aspace_raw` in place during spawn/exec.

**Coupling:** Exec does not create a new `Space`; it mutates the existing one.
The `SpaceId` therefore survives exec, which means a running task cannot
distinguish "the original address space" from "the exec-replaced address space"
by `SpaceId` alone.

**Disposition:** Phase 3 target (see exec-replacement semantics in
`space-object.md`).  For Phase 2, at minimum update `exec.rs` to also update
`space_obj.aspace_raw` when it updates `process.space.aspace_raw`.

---

### B6. `ProcessAddressSpace::from_parts` does not update `space_obj.aspace_raw`

**File:** `kernel/src/task/mod.rs`
**Description:** `ProcessAddressSpace::from_parts` creates a new `space_obj`
with the initial `aspace_raw`.  But if callers later mutate `process.space.aspace_raw`
directly (e.g. during exec), `space_obj.aspace_raw` goes stale.

**Coupling:** `space_obj.aspace_raw` is a snapshot, not a live reference.

**Disposition:** Phase 2 should either:
- Remove `Space.aspace_raw` and make the scheduler read it from
  `process.space.aspace_raw` via the `space_obj`, or
- Add an explicit `Space::set_aspace_raw(u64)` update call in every path that
  writes `process.space.aspace_raw`.

---

### B7. `spawn.rs` `default_process_info` / `inherit_process_info` create `ProcessAddressSpace` inline

**File:** `kernel/src/sched/spawn.rs`
**Description:** Both spawn helpers construct `ProcessAddressSpace::empty()` or
`::from_parts()` inline.  The `space_obj` is created automatically by the
constructor, but callers have no opportunity to provide an externally managed
`Arc<Space>`.

**Coupling:** The spawn path cannot share a `Space` across two newly created
processes (e.g. clone-with-shared-vm is not supported in Phase 1).

**Disposition:** Phase 3 will add an explicit `Arc<Space>` argument to spawn,
allowing the caller to provide an existing space rather than always creating a
new one.

---

### B8. `ProcessAddressSpace` clone behavior not defined

**File:** `kernel/src/task/mod.rs`
**Description:** `ProcessAddressSpace` does not implement `Clone`.  If it were
cloned, it would create a second `space_obj` (with a new `SpaceId`) sharing the
same `mappings` arc — which would be confusing.

**Coupling:** Not currently a problem because `ProcessAddressSpace` is not
cloned anywhere, but the absence of a deliberate policy is a latent trap.

**Disposition:** Add a `#[derive(Clone)]` prohibition note or a custom `Clone`
impl that explicitly creates a *derived* `Space` (new `SpaceId`, same mappings
clone) in Phase 2.

---

### B9. `Thread.mappings` not updated when `space_obj.mappings` changes

**File:** `kernel/src/task/mod.rs`
**Description:** `Thread.mappings` is a cached clone of `process.space.mappings`
set at spawn time.  If `process.space.mappings` is replaced (as during exec),
`Thread.mappings` becomes stale.

**Current behavior:** The exec path replaces the `MappingList` *content* (via
`*ml = MappingList::new()`) rather than replacing the `Arc` itself, so
`Thread.mappings` still points to the same `Arc<Mutex<MappingList>>` and will
see the cleared + repopulated list.

**Coupling:** This works today only because exec replaces the content-under-the-Arc
rather than the Arc itself.  If Phase 3 introduces `Space` replacement (new Arc),
`Thread.mappings` would need to be updated to point to the new `Arc`.

**Disposition:** Phase 3 must update all `Thread.mappings` clones when
replacing the `Space` `Arc`.

---

### B10. `list_processes` snapshot: `space_sharing_count` includes `space_obj` itself

**File:** `kernel/src/sched/mod.rs`
**Description:** `space_sharing_count` is computed as
`Arc::strong_count(&space_obj) − 1`.  The `strong_count` includes references
held by: the `Process` lock-holder, `Thread.mappings` (each thread's cached
arc), and `space_obj` itself.  The reported count therefore over-counts by
the number of threads in the group.

**Coupling:** The count is a best-effort diagnostic value, but its semantics
are confusing.

**Disposition:** Phase 2 should document the precise counting semantics or
introduce a separate counter for "number of tasks sharing this space".

---

### B11. No VFS / procfs exposure of Space

**File:** `kernel/src/vfs/procfs.rs` (or equivalent)
**Description:** The VFS `/proc/<pid>/` tree does not yet expose a `space` file
showing the canonical `Space` representation.

**Disposition:** Phase 2 should add `/proc/<pid>/space` populated via
`kernel::space::bridge::space_from_snapshot`.

---

### B12. `kernel::space::bridge::space_for_current` is untested at runtime

**File:** `kernel/src/space/bridge.rs`
**Description:** `space_for_current` is a correct implementation but has no
unit test because it depends on the scheduler hook infrastructure.

**Disposition:** Phase 2 should add an integration test via the mock-runtime
path (similar to how `authority_for_current` is exercised).

---

## C. Defer Until Later ABI Cleanup (Phase 3+)

These are coupling points that represent the full Unix-compatibility stack.
They are correct to defer because they require broader design decisions.

### C1. `Process` struct as a fusion of all VM + non-VM concerns

**File:** `kernel/src/task/mod.rs`
**Description:** `Process` still combines: lifecycle (`ProcessLifecycle`),
Unix compat (`ProcessUnixCompat`), FD table, namespace, cwd, exec path, and
address space (`ProcessAddressSpace` / `space_obj`).  Only the address-space
subdivision is being extracted in Phase 1.

**Disposition:** Phase 3+ will progressively extract each subdivision into its
own object.  The extraction order is driven by bridge maturity (Job, Authority,
Place, Space, Group in phases 3–5+).

---

### C2. Signal state referencing address space implicitly

**File:** `kernel/src/signal/`
**Description:** Signal handlers and signal stacks (user-mode signal delivery
frames) are written to the user address space using the current task's VM
context.  The signal path never names a `Space` explicitly.

**Disposition:** Phase 3+ will thread `Space` identity into the signal delivery
path.

---

### C3. `SYS_MMAP` / `SYS_MUNMAP` / `SYS_MPROTECT` target implicit current process

**File:** `kernel/src/syscall/handlers/vm.rs`
**Description:** Memory-management syscalls (`mmap`, `munmap`, `mprotect`)
operate on the current task's address space by convention, never accepting an
explicit `Space` handle.

**Disposition:** Phase 3 will introduce explicit `SpaceId`/handle arguments
(gated behind a capability check) enabling cross-process VM operations where
policy permits.

---

### C4. `SYS_SPAWN_PROCESS_EX` creates a new address space internally

**File:** `kernel/src/sched/spawn.rs`
**Description:** The spawn syscall creates a new address space inside the kernel
without any userspace influence over `Space` identity.  Userspace has no way to
request sharing an existing space, creating a COW-derived space, or providing
a pre-built space handle.

**Disposition:** Phase 3+ will expose spawn variants that accept an optional
`Space` handle for thread-group sharing and COW-fork semantics.

---

### C5. fork/clone semantics entirely absent

**Description:** Thing-OS has no `SYS_FORK`; new processes are created via
`SYS_SPAWN_PROCESS[_EX]`.  The spawn path always creates a fresh address space.
There is no mechanism for COW-copy or explicit space derivation.

**Disposition:** Phase 3 design document will define COW-derived Space
semantics and the spawn/clone API surface.

---

### C6. `memfd` / shared anonymous mappings not yet present

**Description:** There is no `memfd_create` equivalent and no shared-memory
import/export mechanism.  All mappings are process-local anonymous or
file-backed.

**Disposition:** After Phase 3 establishes `Space` identity, a `memfd`-backed
mapping type can be introduced that references a shared physical backing from
two different `Space` objects.

---

### C7. No Space-level access control / capability

**Description:** There is no mechanism to grant or restrict another task's
ability to read/write/execute from a given `Space`.  Access is entirely implicit
(you can access your own space; you cannot access others).

**Disposition:** Phase 4+ will define a capability/handle model for `Space`
access.

---

### C8. Debug/procfs `status` file conflates execution and memory identity

**File:** `kernel/src/vfs/procfs.rs`
**Description:** `/proc/<pid>/status` renders execution and memory state in a
Unix-like format without distinguishing `Task`, `Job`, `Space`, `Authority`, and
`Place` concerns.

**Disposition:** Phase 4 will split procfs output into per-object files:
`/proc/<pid>/task`, `/proc/<pid>/job`, `/proc/<pid>/space`, etc., each rendered
via its canonical bridge.

---

## Migration Priority Matrix

| Item | Phase | Urgency    | Effort |
|------|-------|-----------|--------|
| A1: `ProcessAddressSpace` → `Arc<Space>` | Phase 3 | High | Large |
| A2: `ProcessSnapshot` space fields | **Phase 1 ✅** | Critical | Small |
| A3: `thingos::space::Space` type | **Phase 1 ✅** | Critical | Small |
| B1: `Thread.space: Arc<Space>` | Phase 2 | Medium | Medium |
| B2: `Space::activate()` for context switch | Phase 2 | Medium | Medium |
| B3: VM syscalls via `Arc<Space>` | Phase 2 | High | Medium |
| B4: Page fault via `Space::handle_fault` | Phase 2 | Medium | Small |
| B5: Exec as Space replacement | Phase 3 | High | Large |
| B6: `space_obj.aspace_raw` staleness | Phase 2 | Medium | Small |
| B7: Clone-with-shared-vm spawn | Phase 3 | Medium | Large |
| B8: `ProcessAddressSpace` clone policy | Phase 2 | Low | Tiny |
| B9: `Thread.mappings` update on Space replace | Phase 3 | High | Medium |
| B10: `space_sharing_count` semantics | Phase 2 | Low | Tiny |
| B11: `/proc/<pid>/space` | Phase 2 | Low | Small |
| B12: `space_for_current` test | Phase 2 | Low | Small |
| C1: `Process` decomposition | Phase 3+ | High | XLarge |
| C2: Signal path Space threading | Phase 3+ | Medium | Medium |
| C3: Explicit Space in mmap/munmap | Phase 3+ | Medium | Large |
| C4: Spawn with Space handle | Phase 3+ | Medium | Large |
| C5: fork/clone semantics | Phase 3+ | Low | XLarge |
| C6: memfd / shared mappings | Phase 4+ | Low | Large |
| C7: Space capability/access control | Phase 4+ | Low | Large |
| C8: procfs per-object files | Phase 4+ | Low | Medium |

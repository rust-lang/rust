# Namespace Semantics and Roadmap

> **Status:** Namespace isolation is a planned feature. The current kernel uses
> a single shared (global) mount namespace for all processes.  This document
> defines exactly what is and is not isolated today, the intentional API
> contract for stub behaviour, and the staged roadmap for real isolation.

---

## 1. Behaviour Matrix — What Is Global vs Isolated Today

| Resource | Status | Owner | Notes |
|---|---|---|---|
| **File descriptor table** | ✅ Isolated | `Process.fd_table` | Per-process, inherited-by-clone at spawn |
| **Current working directory** | ✅ Isolated | `Process.cwd` | Per-process, inherited at spawn/exec |
| **Environment variables** | ✅ Isolated | `Process.env` | Per-process, inherited at spawn |
| **Virtual memory mappings** | ✅ Isolated | `Process.mappings` | Per-process address space |
| **Mount namespace** | ⚠️ Global stub | `NamespaceRef` (unit struct) | All processes share one global mount table |
| **`SYS_FS_MOUNT` effect** | ⚠️ Global | `vfs::mount::mount()` | A mount made by any process is visible to all |
| **`SYS_FS_UMOUNT` effect** | ⚠️ Global | `vfs::mount::umount()` | An unmount by any process affects all |
| **PID namespace** | ⚠️ Global stub | N/A | Single flat PID space, no nesting |
| **Network namespace** | ⚠️ Not implemented | N/A | No networking stack yet; will be per-namespace when added |
| **IPC namespace** | ⚠️ Global stub | `GLOBAL_HANDLE_TABLE` | Handle table is currently process-local; port IDs are global |

### Key terms

* **Global stub** — the data structure for isolation exists (e.g. `NamespaceRef`
  in `Process.namespace`) but all instances resolve to the same underlying
  shared state.  The field is a placeholder that will gain real semantics in a
  future milestone without changing call sites.

* **Isolated** — each process owns an independent copy; changes in one process
  do not propagate to others.

---

## 2. `NamespaceRef` API Contract

`NamespaceRef` (defined in `kernel/src/vfs/mod.rs`) is the kernel type that
represents a process's view of the VFS mount table.

### Current contract (stub behaviour)

```rust
// kernel/src/vfs/mod.rs
pub struct NamespaceRef; // unit struct — no per-process state yet

impl NamespaceRef {
    /// Returns the single global namespace shared by all processes.
    pub fn global() -> Self { Self }
}
```

* Every call to `NamespaceRef::global()` returns an equivalent value.
* All VFS path resolution ignores `NamespaceRef` and goes directly to the
  global mount table in `vfs::mount`.
* `Process.namespace` is cloned at spawn time via `inherit_process_info` but
  both parent and child receive a copy that resolves to the same global state.

### Behaviour guarantees (valid now and after real isolation lands)

| Guarantee | Rationale |
|---|---|
| Path resolution through `NamespaceRef` is deterministic | Required for correctness |
| `SYS_FS_OPEN` uses the calling process's `namespace` field | Plumbing is in place; no call-site change needed when per-process namespaces are added |
| A process cannot observe a mount it has explicitly unmounted | `umount` removes the entry from whichever table the namespace resolves to |
| `SYS_FS_MOUNT` requires no privilege today (stub) | Will require `CAP_SYS_ADMIN` or namespace ownership in a future milestone |

### Non-guarantees (explicitly not promised today)

* **Isolation**: a mount by process A is immediately visible to process B.
* **Privilege checking**: `SYS_FS_MOUNT` does not verify that the calling
  process has permission to modify the namespace.
* **Snapshot-on-spawn**: spawning a child does not give it a private copy of
  the mount table; both parent and child see subsequent mounts.

---

## 3. Staged Implementation Roadmap

### Milestone NS-1 — Owned mount table (ACT V prerequisite)

**Goal:** `NamespaceRef` contains an `Arc<MountTable>` instead of being a unit
struct.  All processes still share the same `Arc`, so observable behaviour is
unchanged.

**Changes required:**

* Add `MountTable` struct wrapping the current global mount list in
  `kernel/src/vfs/mount.rs`.
* Change `NamespaceRef` to `pub struct NamespaceRef(Arc<MountTable>)`.
* Replace all `NamespaceRef::global()` call sites with a constructor that
  returns the shared singleton.
* VFS resolution helpers (`vfs::path::resolve`) accept `&NamespaceRef` and
  delegate to `namespace.0`.

**Acceptance criteria:**

- [ ] All existing tests pass unchanged.
- [ ] `NamespaceRef::global()` returns a value backed by a real mount table.
- [ ] No change to externally visible syscall behaviour.

---

### Milestone NS-2 — Copy-on-write namespace at spawn

**Goal:** `SYS_SPAWN_PROCESS_EX` can optionally give the child its own private
mount table, initialised as a shallow copy of the parent's.

**Changes required:**

* Add `SpawnFlags::CLONE_NEWNS` (or equivalent field in
  `abi::types::SpawnProcessExReq`) so callers can opt into a private
  namespace.
* In `sched::spawn::inherit_process_info`, branch on the flag: share the
  parent's `Arc<MountTable>` (existing behaviour) or `Arc::new(parent.clone())`
  for a private copy.
* `SYS_FS_MOUNT` / `SYS_FS_UMOUNT` modify `process.namespace.0` (the
  process-local mount table) rather than the global singleton.

**Acceptance criteria:**

- [ ] A process spawned without `CLONE_NEWNS` behaves identically to today.
- [ ] A process spawned with `CLONE_NEWNS` has its own mount table; mounts
  do not propagate back to the parent.
- [ ] Documentation and a unit test cover both cases.

---

### Milestone NS-3 — Privilege enforcement for `SYS_FS_MOUNT`

**Goal:** Only a process that owns a namespace (or holds `CAP_SYS_ADMIN`) may
call `SYS_FS_MOUNT` against it.

**Changes required:**

* Introduce a capability / privilege model (separate issue).
* Add an ownership check in `sys_fs_mount` / `sys_fs_umount`.

**Acceptance criteria:**

- [ ] An unprivileged process receives `EPERM` when trying to mount into the
  global namespace.
- [ ] A process with a private namespace may freely mount within it.

---

### Milestone NS-4 — PID and IPC namespaces

**Goal:** Nested PID spaces (containers) and scoped IPC handle tables.

**Notes:**

* PID namespaces require a mapping layer in `sys_get_tid`, `sys_waitpid`, and
  `/proc`.
* IPC namespace scoping means `GLOBAL_HANDLE_TABLE` becomes per-namespace.
* This milestone is intentionally deferred until NS-2 and NS-3 are stable.

---

## 4. How to Detect the Stub in Code

If you need to guard code that only makes sense with real namespace isolation,
use a feature flag or a runtime check:

```rust
// kernel/src/vfs/mod.rs
impl NamespaceRef {
    /// Returns `true` once per-process namespace isolation is implemented.
    /// Currently always returns `false` (global stub).
    pub fn is_isolated(&self) -> bool {
        false
    }
}
```

Callers that need to behave differently when isolation is active can call
`process.namespace.is_isolated()` and fall back to safe global behaviour when
it returns `false`.

---

## 5. Related Issues and Documents

* **This issue** — Clarify namespace semantics and roadmap
  (stub vs implemented isolation)
* `docs/concepts/process-object.md` — Process / Thread ownership model
* `docs/concepts/janix-guardrails.md` — Architectural guardrails (VFS-first,
  spawn+exec model)
* `kernel/src/vfs/mod.rs` — `NamespaceRef` struct and global mount init
* `kernel/src/vfs/mount.rs` — Global mount table implementation
* `kernel/src/sched/spawn.rs` — `inherit_process_info` (namespace inheritance
  at spawn time)

Follow-up implementation work should be tracked in separate issues linked back
to this document.

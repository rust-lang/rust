# Process Lifecycle

This document describes the full state machine for processes (thread-groups) and
threads in Thing-OS, covering creation, exit, zombie semantics, reaping, and the
rules for parent/child linkage.

---

## Objects

| Object      | Rust type              | Identity        | Purpose                                   |
|-------------|------------------------|-----------------|-------------------------------------------|
| `Process`   | `Arc<Mutex<Process>>`  | PID (= TGID)    | Resource owner: FDs, VM, env, thread list |
| `Thread<R>` | `Box<Thread<R>>`       | TID             | Schedulable execution unit                |

A process is a thread-group.  The thread whose TID equals the PID is the
**thread-group leader**.  All threads in the group share the same `Process` via
`Arc<Mutex<Process>>`.

---

## State machine

```
                   ┌──────────────────────────────────────────────┐
                   │              PROCESS LIFECYCLE               │
                   └──────────────────────────────────────────────┘

    spawn_process / spawn_process_ex
            │
            ▼
      ┌──────────┐   exec (SYS_TASK_EXEC)   ┌──────────┐
      │ Runnable │ ─────────────────────────▶│ Runnable │  (new image)
      └──────────┘                           └──────────┘
           │
           │ scheduler selects
           ▼
      ┌──────────┐
      │ Running  │ ◀──── preemption ────── timer tick
      └──────────┘
           │
           │ sys_exit / group leader exit /
           │ kill_by_tid / SYS_TASK_INTERRUPT
           ▼
      ┌──────────┐     parent calls waitpid    ┌──────────┐
      │  Dead    │ ──────────────────────────▶ │  Reaped  │ (removed from registry)
      │ (zombie) │                             └──────────┘
      └──────────┘
```

Thread states are defined in `kernel/src/sched/state.rs`:

| State       | Meaning                                               |
|-------------|-------------------------------------------------------|
| `Runnable`  | Ready to run, waiting on a CPU                        |
| `Running`   | Currently executing on a CPU                          |
| `Blocked`   | Waiting for an event (I/O, sleep, waiter queue)       |
| `Dead`      | Exited; exit code preserved until reaped by parent    |

---

## Parent/child linkage

- Every `Process` records its parent's PID in `Process.ppid`.
- The parent PID is set at spawn time and does not change.
- `waitpid` filters children by matching `child.ppid == parent.pid`.

**Reparenting**: Not yet implemented.  If a parent exits before its children
those children currently become un-waitable (ECHILD from any waitpid call on
them by a different process).  A future act will reparent to `init` (PID 1).

---

## Exit / Dead transition

### Single thread (non-leader)

1. Thread calls `sys_exit(code)` → `exit::<R>(code)` → `terminate_current`.
2. `mark_task_exited::<R>(sched, tid, code)` is called:
   - `Thread.state` set to `Dead`.
   - `Thread.exit_code` set to `Some(code)`.
   - TID removed from `Process.thread_ids`.
   - All `exit_waiters` are drained and woken.

### Thread-group leader exit

When the exiting TID equals `Process.pid`:
- All remaining TIDs in `Process.thread_ids` are also marked `Dead` with the
  same exit code (thread-group exit).
- `Process.thread_ids` is drained to empty.

### exec collapse

`SYS_TASK_EXEC` (= `task_exec_current`) replaces the calling process image
in five phases.  Each phase has explicit invariants.

**Phase 1 – Atomic gate**
- `Process.exec_in_progress = true`.
- Any concurrent `SYS_SPAWN_THREAD` call for this process now returns `EAGAIN`.
- If the exec fails before Phase 5 the flag is cleared (`false`) and the
  process/thread-group is left intact (rollback path).

**Phase 2 – Sibling collection**
- All TIDs in `Process.thread_ids` except the exec-caller are collected into a
  local list.  The caller's own TID is excluded.

**Phase 3 – Thread-group collapse (deterministic)**
- Every sibling TID collected in Phase 2 is killed via `kill_by_tid_current`
  → `mark_task_exited`.
- `mark_task_exited` removes each killed TID from `Process.thread_ids` and
  sets its `Task.state = Dead` in the registry.
- After Phase 3 completes, `Process.thread_ids` contains **exactly one entry**:
  the exec-caller TID.  This is a hard invariant — the commit phase (Phase 5)
  must never be reached with any sibling still alive.

**Phase 4 – Image swap**
- The new ELF image is read, loaded into a fresh address space, and the
  auxiliary vector is rebuilt.
- `Process.mappings` is replaced with the new mapping list.
- `Task.aspace` and `Process.aspace_raw` are updated to the new page-table root.
- `Process.fd_table.close_on_exec()` is called: FDs with `FD_CLOEXEC` are closed.
- All pre-exec metadata fields are overwritten:
  - `Process.argv` ← new argument list
  - `Process.env`  ← new environment
  - `Process.auxv` ← rebuilt from the new image's ELF headers
  - `Process.exec_path` ← path to the new binary

**Phase 5 – Commit**
- `Process.exec_in_progress = false`.
- The exec-caller's `Task.ctx` is replaced with a new user context at the new
  image's entry point.
- Control is transferred to the new image via `switch_with_tls`.  This call
  must **never** return to the exec call site.

**No stale metadata invariant**
After Phase 5, no field that was set before `SYS_TASK_EXEC` was called
remains visible through `Process`:
- Pre-exec `argv`, `env`, `auxv`, `exec_path` are replaced.
- Pre-exec mappings and address-space token are replaced.
- `FD_CLOEXEC`-flagged file descriptors are closed.
- `exec_in_progress` is cleared.

Implementation: `task_exec_current` in `kernel/src/task/exec.rs`.

---

## Zombie semantics and reaping

A thread in the `Dead` state is called a **zombie**: it has exited but its
registry entry is still present so the parent can collect the exit status.

The exit record is consumed by `waitpid`:

```
parent: waitpid(child_pid, &status, 0)
  → finds child with state == Dead
  → copies exit_code → status
  → removes child's Thread<R> from ThreadRegistry   ← reaping
  → returns child_pid
```

Key invariants:
- A zombie entry persists **until** the parent (or a future init-reaper) calls
  `waitpid` and collects the status.
- After reaping the TID is gone from the registry.  A second `waitpid` for the
  same child returns `ECHILD`.
- `WNOHANG` does **not** reap: it returns `(0, 0)` immediately without removing
  any live child record.

Implementation: `waitpid_for_pid` in `kernel/src/sched/mod.rs`.

---

## waitpid / reaping rules

| `pid` argument | Meaning                                          |
|----------------|--------------------------------------------------|
| `pid > 0`      | Wait for the specific child with that PID        |
| `pid == -1`    | Wait for any child of the calling process        |
| `pid == 0`     | Same as `pid == -1` (process-group not tracked)  |

| `flags`  | Meaning                                                    |
|----------|------------------------------------------------------------|
| `0`      | Block until a child exits                                  |
| `WNOHANG`| Return `(0, 0)` immediately if no child has already exited |

Return values:
- `Ok((child_pid, exit_code))` — exit status collected and child reaped.
- `Err(ECHILD)` — no matching children exist (all already reaped or never existed).

---

## Thread-group leader and group-exit rules

- The thread-group leader is the thread whose **TID == PID**.
- When the leader exits, all remaining sibling threads are killed atomically
  (same exit code).
- Non-leader threads may exit individually; they are removed from
  `Process.thread_ids` but do not affect siblings.
- A `waitpid` in the parent waits for the **process** (thread-group leader TID).

---

## No-orphan guarantee

After a successful `waitpid` returns:
- The child's `Thread<R>` box is removed from `ThreadRegistry`.
- No further reference to it exists (the `Box` is dropped).
- Repeated `waitpid` returns `ECHILD`.

This ensures that process records do not accumulate indefinitely even when
processes exit frequently.

---

## Acceptance criteria

- [x] Parent/child linkage via `Process.ppid` set at spawn time
- [x] Exit state transitions: `Dead` on `sys_exit` / leader exit / kill
- [x] Thread-group leader exit kills remaining siblings (group exit)
- [x] Zombie semantics: Dead record persists until parent reaps
- [x] Reaping: `waitpid` removes the dead child from the registry
- [x] No orphaned records after `waitpid` success path
- [x] `WNOHANG`: returns `(0, 0)` without reaping live children
- [x] `ECHILD` after reaping (second `waitpid` for same child)
- [x] exec collapse: sibling threads killed deterministically; exec-caller is sole survivor
- [x] exec invariants documented (see "exec collapse" section above)
- [x] exec invariants tested (`kernel/src/task/exec.rs` and `kernel/src/sched/mod.rs`)
- [x] No stale pre-exec process metadata (argv, env, auxv, exec_path, mappings, aspace_raw) visible after successful exec
- [ ] Reparenting to init (PID 1) when parent exits (future act)

---

## Related documents

- `docs/concepts/process-object.md` — `Process` / `Thread<R>` struct design
- `docs/concepts/scheduling.md` — scheduler state and preemption
- `docs/concepts/thingos-guardrails.md` — architecture guardrails
- `abi/src/numbers.rs` — syscall numbers (`SYS_WAITPID`, `SYS_SPAWN_PROCESS`, …)

# Process Object Design

## Overview

Thing-OS distinguishes two first-class kernel objects:

| Object      | Rust type       | Purpose                                |
|-------------|-----------------|----------------------------------------|
| `Thread<R>` | scheduler entry | schedulable execution unit             |
| `Process`   | resource owner  | PID, FDs, VM mappings, env, thread IDs |

A `Thread<R>` always belongs to exactly one `Process` (via
`Arc<Mutex<Process>>`).  Pure kernel threads have `process_info = None`.

## Module layout

```
kernel/src/task/mod.rs      – Thread<R> struct, Process struct, type aliases
kernel/src/task/registry.rs – ThreadRegistry<R>, get_thread / get_thread_mut
kernel/src/sched/state.rs   – ThreadId, ThreadState, ThreadPriority, ThreadSchedFields
kernel/src/sched/spawn.rs   – spawn, spawn_user_thread, spawn_user_task_full
kernel/src/task/exec.rs     – exec/exec_current syscall implementation
```

## Naming convention

| New (preferred)         | Old alias (backward compat)  |
|-------------------------|------------------------------|
| `Thread<R>`             | `type Task<R> = Thread<R>`   |
| `Process`               | `type ProcessInfo = Process` |
| `ThreadId`              | `type TaskId = ThreadId`     |
| `ThreadState`           | `type TaskState = ThreadState`|
| `ThreadPriority`        | `type TaskPriority = ThreadPriority` |
| `ThreadSchedFields`     | `type TaskSchedFields = ThreadSchedFields` |
| `ThreadRegistry<R>`     | `type TaskRegistry<R> = ThreadRegistry<R>` |
| `get_thread`            | `get_task` (forwarding fn)   |
| `get_thread_mut`        | `get_task_mut` (forwarding fn)|

All backward-compat aliases are defined in the same module as the primary type.
Prefer the new `Thread`/`ThreadId`/… names in all new code.

## VM ownership

`Process.mappings` is an `Arc<Mutex<MappingList>>` that is the authoritative
owner of the process's VM mapping list.  At thread-creation time the same
`Arc` is **cloned** into `Thread.mappings` so the scheduler's per-CPU
`CURRENT_MAPPINGS` cache can be updated on context switches without locking
the `Process` mutex.

```
Process.mappings ──(Arc clone)──► Thread.mappings  (for every thread)
                 ──(Arc clone)──► Thread.mappings
                 └── same underlying MappingList object
```

Any mutation of the mapping list goes through either pointer; both see the
same data because they share the same allocation.

## Locking rules

1. The `Process` mutex (`Arc<Mutex<Process>>`) must **never** be acquired while
   the global scheduler lock (`SCHEDULER.lock()`) is held.
2. The `ThreadRegistry` (guarded by `REGISTRY`) serialises access to all
   `Thread<R>` objects; IRQs are disabled while the guard is held.
3. A thread may lock its own `Process` outside of IRQ-disabled sections.

## Thread lifecycle

```
spawn_user_task_full / spawn_process
  └─ Scheduler::spawn_user_task()    – allocates Thread, Process, MappingList
       └─ default_process_info()     – creates Process with shared mappings Arc
  └─ stores in ThreadRegistry

spawn_user_thread / SYS_SPAWN_THREAD
  └─ Scheduler::spawn_user_thread()  – allocates Thread
       └─ inherits mappings Arc from parent Thread.mappings
       └─ adds TID to Process.thread_ids

thread exit / mark_task_exited()
  └─ ThreadState::Dead set in registry and sched state
  └─ TID removed from Process.thread_ids
  └─ if TID == PID (group leader): remaining siblings killed

exec / task_exec_current()
  └─ Process.exec_in_progress = true
  └─ sibling threads killed, Process.thread_ids drained to [caller]
  └─ Process reset with new argv/env/auxv/exec_path
  └─ Process.exec_in_progress = false
```

## Acceptance criteria (issues #734 / #736)

- [x] `Process` is the unit of resource ownership (FDs, mappings, env, CWD)
- [x] `Thread<R>` is the explicit schedulable object; `Task<R>` is an alias
- [x] `Thread.mappings` is always a clone of `Process.mappings` (same Arc)
- [x] Thread creation inherits the parent's `Process.mappings` Arc
- [x] `Process.thread_ids` tracks all live TIDs in the group
- [x] Backward-compat aliases keep existing code compiling without changes
- [ ] Per-process namespace divergence — see `docs/concepts/namespaces.md` for roadmap
- [ ] `ThreadId` exposed separately from `ProcessId` in ABI types (future)

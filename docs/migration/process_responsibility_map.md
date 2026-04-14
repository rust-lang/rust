# Process Responsibility Map (Migration Inventory)

> **Status**: Active migration control document — Phase 8 baseline.
> Space-oriented subdivision introduced (issue: Isolate Address Space into Space-Oriented Substructure).
> Cite this document in all subsequent decomposition issues.

---

## Purpose

This document is the canonical planning reference for the incremental decomposition
of the `Process` struct in `kernel/src/task/mod.rs`.

`Process` is the current transitional resource-ownership object for user processes.
It carries responsibilities that properly belong to several future first-class
concepts: **Task**, **Job**, **Group**, **Authority**, **Place**, and **Space** (plus
a legacy-compatibility quarantine for Unix baggage that does not map cleanly onto
any of those).

The goal is to make every responsibility explicit — what it is, where it lives
today, where it is going, and in what order to move it — so that migration work
is systematic rather than ad hoc.

---

## Object glossary

| Future concept | Rough meaning                                             | Bridge module (today)           |
|----------------|-----------------------------------------------------------|---------------------------------|
| `Task`         | Schedulable execution unit (≈ thread view)                | `kernel::task::bridge`          |
| `Job`          | Lifecycle/accounting container (creation → exit → reap)   | `kernel::job::bridge`           |
| `Group`        | Coordination domain (process group, session)              | `kernel::group::bridge`         |
| `Authority`    | Permission/credential context                             | `kernel::authority::bridge`     |
| `Place`        | World/context boundary (cwd, namespace, root)             | `kernel::place::bridge`         |
| `Space`        | Address space and memory-mapping ownership                | *(not yet introduced)*          |
| Handle table   | Open-file/resource descriptor table                       | *(not yet introduced)*          |
| Spawn record   | Immutable invocation context (argv, auxv, exec_path)      | *(not yet introduced)*          |
| Legacy compat  | Unix baggage with no clean architectural home             | quarantined inside `Process`    |

---

## Responsibility inventory

The table below covers every field currently present in `Process`
(`kernel/src/task/mod.rs`) plus the responsibility-adjacent fields in
`Thread<R>` that have a strong lifecycle coupling to `Process`.

### Process fields

| Current field / concept             | Current role            | Future owner     | Migration status   | Notes                                                                                     |
|-------------------------------------|-------------------------|------------------|--------------------|-------------------------------------------------------------------------------------------|
| `pid: u32` (TGID)                   | Process identity        | Job + Space      | bridge in place    | PID doubles as TGID; future `Job` carries lifecycle ID, `Space` carries VM identity.     |
| `ppid: u32`                         | Parent/child linkage    | Job              | bridge in place    | Needed for `waitpid` parent filter; migrates with wait semantics.                         |
| `thread_ids: Vec<ThreadId>`         | Thread membership list  | Job + Task       | bridge in place    | Group-leader exit and exec collapse both rely on this; migrates with Job lifecycle.       |
| `exec_in_progress: bool`            | Exec synchronisation    | Job              | extract next       | Guards concurrent `SYS_SPAWN_THREAD` during exec; belongs with Job lifecycle gate.        |
| `children_done: VecDeque<(u32,i32)>`| Waitpid exit queue      | Job              | extract next       | Exit-status accumulator for parent `waitpid`; should move with Job wait semantics.       |
| `mappings: Arc<Mutex<MappingList>>` | VM mapping ownership    | Space            | **grouped** ✓      | Moved into `ProcessAddressSpace` subdivision (`Process.space.mappings`). Extract `Space` next. |
| `aspace_raw: u64`                   | Address-space token     | Space            | **grouped** ✓      | Moved into `ProcessAddressSpace` subdivision (`Process.space.aspace_raw`). Extract `Space` next.|
| `fd_table: FdTable`                 | Open-file resource table| Handle table     | keep for now       | No handle-table concept yet; extract after Authority stabilises (Phase 9+).               |
| `cwd: String`                       | Current working dir     | Place            | **bridged** ✓      | Surfaced through `kernel::place::bridge::place_from_snapshot`; field stays as backing.   |
| `namespace: NamespaceRef`           | VFS mount-table view    | Place            | bridge in place    | Unit struct today (global); per-process isolation deferred. Provisional backing for Place.|
| `exec_path: String`                 | Executable identity     | Authority        | **bridged** ✓      | Used as `Authority::name` fallback in `kernel::authority::bridge`; field stays as backing.|
| `pgid: u32`                         | Process group ID        | Group            | **bridged** ✓      | Surfaced through `kernel::group::bridge`; raw field is quarantined Unix state.            |
| `sid: u32`                          | Session ID              | Group            | **bridged** ✓      | Surfaced through `kernel::group::bridge`; raw field is quarantined Unix state.            |
| `session_leader: bool`              | TTY foreground proxy    | Group            | **bridged** ✓      | Used as `GroupKind::Foreground` heuristic until `foreground_pgid` is queryable.           |
| `signals: ProcessSignals`           | Signal dispositions + pending set + stop/alarm state | Legacy compat → Authority/Group | quarantine | SIGTTOU/SIGTTIN job-control is Group concern; disposition table is Authority concern; not yet split. |
| `argv: Vec<Vec<u8>>`                | Spawn-time arg vector   | Spawn record     | quarantine         | No principled spawn-record concept yet; quarantined legacy compat.                        |
| `env: BTreeMap<Vec<u8>,Vec<u8>>`    | Unix environment blob   | Legacy compat    | quarantine         | Raw key→value env map has no clean architectural home; quarantined Unix baggage. Not part of any future canonical concept until a principled env-passing design is adopted. |
| `auxv: Vec<(u64,u64)>`              | ELF auxiliary vector    | Spawn record     | quarantine         | ELF-specific Unix compat; quarantined until a spawn-record concept exists.                |

### Thread fields with Process-level coupling

| Current field / concept             | Current role            | Future owner     | Migration status   | Notes                                                                                     |
|-------------------------------------|-------------------------|------------------|--------------------|-------------------------------------------------------------------------------------------|
| `exit_code: Option<i32>`            | Exit state              | Job              | **bridged** ✓      | Read by `kernel::job::bridge::job_exit_from_snapshot`; migrates with Job exit semantics. |
| `exit_waiters: WaitQueue`           | Per-thread exit waiter  | Job / Task       | bridge in place    | Level-triggered wait queue; logically a Job wait mechanism, housed in Thread today.       |
| `signals: ThreadSignals`            | Per-thread signal mask + pending | Legacy compat → Authority | quarantine | Thread-directed signal delivery; quarantined Unix compat alongside ProcessSignals. |
| `process_info: Option<Arc<Mutex<Process>>>` | Thread → process back-reference | Task → Job ref | keep for now | Every user thread holds this to reach shared resources; shrinks as fields migrate out. |

---

## What is already bridged

The following bridges are in place (Phase 3–8). Each bridge is the **canonical
public surface** for its domain; new code must go through the bridge, not read
`Process` fields directly.

| Bridge module                       | What it maps                                        | Phase |
|-------------------------------------|-----------------------------------------------------|-------|
| `kernel::task::bridge`              | `ThreadState` → `thingos::task::Task`               | pre-3 |
| `kernel::job::bridge`               | `Process`+`Thread` lifecycle → `thingos::job::Job`  | 3     |
| `kernel::group::bridge`             | `pgid`/`sid`/`session_leader` → `thingos::group::Group` | 4 |
| `kernel::authority::bridge`         | `name`/`exec_path` → `thingos::authority::Authority` | 7   |
| `kernel::place::bridge`             | `cwd`/`namespace` → `thingos::place::Place`         | 8     |

---

## What is not yet bridged (nor extracted)

| Responsibility                                        | Blocker / note                                                   |
|-------------------------------------------------------|------------------------------------------------------------------|
| Address space / VM mappings (`space.mappings`, `space.aspace_raw`) | Now grouped under `ProcessAddressSpace` inside `Process.space`. Future work: introduce first-class `Space` object, promote subdivision into it. |
| FD / handle table (`fd_table`)                        | Handle-table concept not yet introduced; extract after Phase 9+. |
| Spawn invocation context (`argv`, `env`, `auxv`)      | No spawn-record concept; quarantined until one exists.           |
| Signal state (`ProcessSignals`, `ThreadSignals`)      | Needs split: disposition → Authority, job-control → Group; complex. |
| UID/GID / capability mask *(planned addition)*        | Not yet present in `Process`; must be added to `Process` before a full Authority extraction is possible. Tracked here as a prerequisite gap, not as a current field. |
| Controlling terminal / TTY attachment                 | Belongs to `Presence` (not yet introduced); quarantined for now. |
| Per-process namespace isolation                       | `NamespaceRef` is a unit struct; defer until namespace work.     |
| Reparenting to init (orphan reaping)                  | No init-process concept yet; deferred.                           |

---

## Suggested extraction order

The ordering below minimises cascading breakage. Each step should be a
self-contained issue that cites this document.

| Step | Responsibility                         | Destination    | Rationale                                                   |
|------|----------------------------------------|----------------|-------------------------------------------------------------|
| 1    | Exit state + wait queue + children_done| Job            | Already bridged; lowest-risk first extraction.              |
| 2    | exec_in_progress lifecycle gate        | Job            | Small flag that belongs with Job lifecycle; extract with step 1. |
| 3    | thread_ids membership                  | Job + Task     | Needed for group-exit; extract after Job lifecycle is solid. |
| 4    | pgid / sid / session_leader (raw fields)| Group         | Bridges exist; can quarantine raw fields once Group carries truth. |
| 5    | Signal dispositions (ProcessSignals)   | Authority      | Disposition table is permission context; needs Authority stabilised first. |
| 6    | Job-control stop signals (in signals)  | Group          | SIGSTOP/SIGCONT/SIGTTOU semantics belong in Group; extract after step 4. |
| 7    | `Process.space` → first-class `Space`  | Space          | `ProcessAddressSpace` subdivision is now the extraction seam. Promote `Process.space` into a first-class `Space` kernel object; share across threads/processes as needed. |
| 8    | fd_table                               | Handle table   | Introduce handle-table concept; extract after Space.        |
| 9    | cwd / namespace / root                 | Place          | Bridge exists; promote backing fields into Place substructure after Space. |
| 10   | argv / env / auxv                      | Spawn record   | Introduce spawn-record concept; then move quarantined fields.|
| 11   | UID/GID / capabilities                 | Authority      | Add fields to Process first; then surface through authority bridge.|
| 12   | Controlling terminal / TTY             | Presence       | Introduce Presence; then extract from signals/session state. |

---

## Migration status legend

| Status            | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| **bridged** ✓     | A bridge module exists and is the canonical public surface.      |
| **grouped** ✓     | Fields are grouped into a named subdivision (e.g. `ProcessAddressSpace`) as an extraction seam. Next step: promote to first-class object. |
| bridge in place   | Bridge exists but raw field still serves as the only backing.    |
| extract next      | Low-risk; extraction is the next logical step for this concern.  |
| keep for now      | No extraction work until a prerequisite concept is introduced.   |
| quarantine        | Unix baggage; isolated, not yet removable, future removal planned.|
| remove later      | Will be deleted once extraction is complete and all callers updated.|

---

## Related documents

- `docs/concepts/process-object.md` — `Process` / `Thread<R>` struct design
- `docs/concepts/process-lifecycle.md` — state machine, exec collapse, zombie semantics
- `docs/concepts/janix-guardrails.md` — architecture guardrails for all kernel changes
- `kernel/src/task/mod.rs` — primary `Process` struct (inline migration annotations); `ProcessAddressSpace` subdivision
- `kernel/src/job/bridge.rs` — Phase 3 lifecycle bridge
- `kernel/src/group/bridge.rs` — Phase 4 coordination bridge
- `kernel/src/authority/bridge.rs` — Phase 7 permission bridge
- `kernel/src/place/bridge.rs` — Phase 8 world-context bridge
- `thingos/src/job.rs` — canonical `Job` / `JobState` / `JobExit` types

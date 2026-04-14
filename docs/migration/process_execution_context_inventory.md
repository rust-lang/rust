# Process-Carried Execution Context Inventory

> **Status**: Phase 9 baseline inventory — initial mapping artifact.
> This document answers: *"What does `Process` currently carry that tells the system
> where execution is happening?"*
> It is observational, not aspirational.  Do not begin extractions until a
> dedicated issue explicitly scopes that work.
>
> Companion documents:
> - `docs/migration/process_responsibility_map.md` — full field-ownership decomposition
> - `docs/migration/authority_inventory.md` — credential/permission inventory

---

## Purpose

`Process` in Thing-OS (`kernel/src/task/mod.rs`) implicitly owns a large slice
of *execution context* — the ambient state that answers:

> *"In what world does this execution happen, and what are its context
> boundaries?"*

ThingOS is moving away from the Unix pattern where a process silently inherits
and carries all of that state.  The emerging model redistributes these concerns
across **Place** (world/context/visibility), **Presence** (person-in-place
embodiment), **Group** (shared coordination/control), and a **Legacy
compatibility** quarantine for Unix session/TTY baggage that has no clean
architectural home yet.

This document:

1. Enumerates every execution-context field and implicit assumption currently
   attached to `Process` (or its closely-coupled subsystems).
2. Assigns each item an intended future owner.
3. Records a migration status so this file can act as an active planning
   artifact rather than a static snapshot.
4. Highlights likely extraction seams for future Place/Presence work.

---

## Sources consulted

| File | What it contains |
|------|-----------------|
| `kernel/src/task/mod.rs` | `Process`, `ProcessUnixCompat`, `ProcessLifecycle`, `ProcessAddressSpace`; inline migration annotations |
| `kernel/src/place/bridge.rs` | Phase 8 Place bridge — canonical world-context surface |
| `kernel/src/group/bridge.rs` | Phase 4 Group bridge — session/pgid coordination |
| `kernel/src/vfs/devfs.rs` | `ConsoleTtyState`, `ConsoleCaller`, job-control enforcement |
| `kernel/src/sched/hooks.rs` | `ProcessSnapshot` — snapshot fields that back procfs and bridges |
| `kernel/src/signal/mod.rs` | `ProcessSignals`, `ThreadSignals` — SIGTTOU/SIGTTIN job-control coupling |
| `kernel/src/syscall/handlers/vfs.rs` | `sys_fs_getcwd`, `sys_fs_chdir`, path resolution |
| `kernel/src/syscall/handlers/process.rs` | spawn, exec, kill, setsid, setpgid |
| `stem/src/syscall/vfs.rs` | `tcgetpgrp`, `tcsetpgrp`, `tcgetattr`, `tcsetattr` |
| `thingos/src/place.rs` | Canonical `Place` struct (cwd, namespace, root) |
| `docs/migration/process_responsibility_map.md` | Pre-existing ownership map |

---

## Concept glossary

| Concept | Meaning in the future model |
|---------|----------------------------|
| **Place** | World-context and visibility boundary — owns cwd, namespace, and filesystem root |
| **Presence** | Person-in-place embodiment and active terminal/UI participation |
| **Group** | Shared coordination domain — process groups, sessions, foreground/background control |
| **Legacy compat** | Transitional Unix session, TTY, and environment baggage with no clean future home yet |
| **Spawn record** | Immutable invocation context (argv, auxv, exec\_path) — not world-context |

---

## 1. Execution-Context Inventory Table

The table below covers every field, subsystem, and behavioral assumption that
currently tells the system *where* or *in what world* a process is executing.
Items are grouped by location.

### 1.1 Direct `Process` fields (`kernel/src/task/mod.rs`)

| Current Process-Carried Context | Field / path | Current role | Future owner | Migration status | Notes |
|---|---|---|---|---|---|
| Current working directory | `Process.cwd` | Per-process path base for relative resolution | **Place** | **bridged ✓** | Surfaced through `kernel::place::bridge::place_from_snapshot` → `Place::cwd`; raw field stays as transitional backing. Extraction seam: move into `Place`-shaped substructure and replace raw String with VFS-node ref. |
| VFS namespace / mount-table view | `Process.namespace` (NamespaceRef) | Determines which mount table the process sees | **Place** | bridge in place | `NamespaceRef` is a unit struct today — all processes share one global mount table. Bridged as `Place::namespace = "global"`. Per-process isolation deferred. |
| Effective filesystem root | *(no field yet)* | Should bound the process's visible VFS tree | **Place** | not yet added | `Place::root` is hardcoded to `"/"` in the bridge. No per-process chroot/pivot-root implemented. Add field to `Process` before extraction can begin. |
| Executable image path | `Process.exec_path` | Path of the running image | Authority (name fallback) | **bridged ✓** | Feeds `Authority::name` in `kernel::authority::bridge`. Execution-context dimension: identifies *which world-image* is running. Not cwd/namespace. |
| File descriptor table | `Process.fd_table` | Open-file "window" into the VFS | Handle table → Place adjacency | keep for now | FDs 0/1/2 carry stdout/stdin/stderr and may be TTY FDs. Constitutes the process's concrete VFS interface. No handle-table concept yet; extract after Authority stabilises. |

### 1.2 `ProcessUnixCompat` fields — Unix execution context baggage

These fields live in `Process.unix_compat` behind an explicit legacy boundary.
They are quarantined because they mix world-context, coordination, and
invocation history with no clean extraction path yet.

| Current Process-Carried Context | Field / path | Current role | Future owner | Migration status | Notes |
|---|---|---|---|---|---|
| Process group ID | `Process.unix_compat.pgid` | Foreground/background coordination domain | **Group** | **bridged ✓** | Surfaced through `kernel::group::bridge`. Governs `kill(0/neg)` and job-control signal routing. Raw field quarantined. |
| Session ID | `Process.unix_compat.sid` | Unix login/control context | **Group / legacy** | quarantine | Used by `group::bridge` and TTY job-control. No formal session object. Do not deepen. |
| Session leader flag | `Process.unix_compat.session_leader` | TTY foreground ownership proxy | **Group / Presence / legacy** | quarantine | Used as a heuristic by `group::bridge` because `ConsoleTtyState::foreground_pgid` is not queryable outside `devfs`. Imprecise: background session leaders mis-report. |
| Environment variable map | `Process.unix_compat.env` | Inherited KEY=VALUE env blob | **Place / legacy** | quarantine | Affects PATH, HOME, etc. — classic world-context. No clean Place-facing env model yet; quarantined Unix baggage. Do not deepen. |
| Argument vector | `Process.unix_compat.argv` | Spawn-time arguments | Spawn record | quarantine | Invocation context, not world-context. Quarantined until a spawn-record concept exists. |
| ELF auxiliary vector | `Process.unix_compat.auxv` | ELF AT\_\* entries (layout, vDSO, uid/gid stubs) | Spawn record | quarantine | ELF-specific Unix compat. AT\_UID/AT\_GID are zeros (no uid/gid model yet). |
| Signal state (job-control portion) | `Process.unix_compat.signals` (SIGTTOU, SIGTTIN, SIGSTOP) | Job-control stop/continue | **Group / Presence** | quarantine | SIGTTOU/SIGTTIN enforce background-process TTY access policy. SIGSTOP/SIGCONT are coordination signals. Entangled with full signal table; split requires signal authority work first. |
| Typed message inbox | `Process.unix_compat.message_inbox` | Process-local bounded inbox | Group broadcast / IPC | keep for now | Not world-context per se; delivery coordination. Kept here pending a Group-broadcast or IPC inbox model. |

### 1.3 TTY / Console device state (implicit process execution context)

These fields live in `kernel/src/vfs/devfs.rs` as global TTY device state.
They are not fields on `Process` but they *define the execution context* for
processes that hold or want the controlling terminal.

| Current Process-Carried Context | Location | Current role | Future owner | Migration status | Notes |
|---|---|---|---|---|---|
| Controlling session (`controlling_sid`) | `ConsoleTtyState` (global in devfs) | Which session "owns" `/dev/console` | **Group / Presence** | quarantine | Acquired implicitly on first open by a session leader. No explicit attach/detach protocol. Belongs to Presence once that concept exists. |
| Foreground process group (`foreground_pgid`) | `ConsoleTtyState` (global in devfs) | Which pgid has TTY foreground | **Group** | quarantine | Set via `TIOCSPGRP` / `tcsetpgrp`. Guards SIGTTOU/SIGTTIN delivery. Needed by `group::bridge` but not queryable outside `devfs` today. |
| Terminal mode settings (`termios`) | `ConsoleTtyState` (global in devfs) | Input/output processing flags | **Place / Presence** | keep for now | Affects how the TTY processes I/O. Shared across all sessions today (single console). Per-session termios is a future Presence concern. |
| Console caller context (`ConsoleCaller`) | Derived on each VFS call | sid + pgid + session\_leader used for job-control enforcement | **Group / Presence** | quarantine | Constructed transiently from `Process` fields. Extraction seam: replace with a Group membership query once Group is a first-class object. |

### 1.4 `Thread<R>` fields with execution-context coupling

| Current Process-Carried Context | Field / path | Current role | Future owner | Migration status | Notes |
|---|---|---|---|---|---|
| TLS base pointer | `Thread.user_fs_base` | User-mode FS segment (thread-local storage) | Task / Space | keep for now | Defines the thread's per-CPU data window. Architecture-specific. Not world-context but shapes *execution* context. |
| Per-thread signal mask | `Thread.signals` (ThreadSignals) | Blocked signal set + thread-directed pending | Legacy compat → Authority | quarantine | Masks interrupts to the thread's execution. Extraction blocked on signal authority work. |
| Kernel/user mode | `Thread.is_user` | Distinguishes kernel from user execution context | Task | keep for now | Implicit trust boundary; kernel threads bypass VFS auth. Architectural; not migrating. |
| Process back-reference | `Thread.process_info` | Reaches shared Process resources from thread | Task → Job ref | keep for now | Every user thread holds this; shrinks as fields migrate out of `Process`. |

### 1.5 Inherited / behavioral execution-context assumptions

These are not single fields but cross-cutting behavioral assumptions that
propagate execution context across fork/exec/spawn.

| Current Process-Carried Context | Where assumed | Future owner | Migration status | Notes |
|---|---|---|---|---|
| cwd inheritance across spawn | `spawn_process_ex` copies `cwd` from parent unless overridden by `cwd_ptr` | **Place** | extract next | `spawn_process_ex_cwd` allows override; should be the default spawn path. Extraction seam: new processes derive cwd from a Place descriptor, not raw String copy. |
| env inheritance across spawn | `ProcessUnixCompat::inherit` copies `env` from parent | **Place / legacy** | quarantine | Classic Unix env propagation. Treat as legacy until a principled env-passing model is designed. |
| pgid/sid inheritance across spawn | `ProcessUnixCompat::inherit` copies `pgid`/`sid` | **Group** | quarantine | Session membership propagates implicitly. Should become explicit Group membership in the future model. |
| Implicit TTY acquisition | `ConsoleNode::maybe_acquire_controlling_tty` auto-claims on first open by session leader | **Presence / Group** | quarantine | No explicit "attach to terminal" syscall. Acquisition is a side effect of `open("/dev/console")` by a session leader. Extraction seam: introduce explicit terminal-attach operation. |
| Relative path resolution against cwd | `resolve_path` in VFS handlers reads `Process.cwd` | **Place** | **bridged ✓** | All relative paths resolve through `Process.cwd`. New code must not read `cwd` directly; use `place::bridge` instead. |
| FD 0/1/2 as TTY attachment | FD table at spawn time | **Place / Presence** | keep for now | stdin/stdout/stderr are the process's concrete I/O world. If they point to a TTY, that is implicit Presence. No abstraction yet. |
| Session lifecycle via `setsid` | `sys_signal_setsid` | **Group / Presence** | quarantine | Creates a new session and breaks TTY attachment. Architectural seam for a future Presence detach + Group create operation. |

---

## 2. Extraction Seams Summary

The table below highlights the most obvious next cuts, ordered approximately
by extraction difficulty.

| Seam | Source | Target | Why it is the natural cut |
|------|--------|--------|--------------------------|
| cwd / namespace / root | `Process.cwd`, `Process.namespace`, *(no root field)* | Place | Bridge already exists. Raw backing fields are the only remaining coupling. Next step: replace raw String with VFS-node reference; add root field; promote to Place substructure. |
| Explicit cwd override at spawn | `spawn_process_ex` `cwd_ptr` | Place | Already supported via `spawn_process_ex_cwd`; making it the standard spawn path decouples child Place from parent cwd. |
| Env blob → Place-facing env model | `Process.unix_compat.env` | Place / legacy | env directly controls world visibility (PATH, HOME, locale). Needs a Place-facing design before extraction. Quarantine until then. |
| Session/TTY glue | `sid`, `session_leader`, `ConsoleTtyState` | Group / Presence | Natural boundary once Group becomes first-class. Requires explicit terminal-attach semantics (Presence) before `maybe_acquire_controlling_tty` can be removed. |
| Foreground pgid query outside devfs | `ConsoleTtyState::foreground_pgid` | Group bridge | Unblocks `group::bridge` from using heuristic; enables precise Foreground/Background determination. Low-scope change; do not deepen the global TTY state. |
| Implicit TTY acquisition → explicit attach | `ConsoleNode::maybe_acquire_controlling_tty` | Presence | Replace side-effect open with an explicit `SYS_TTY_ATTACH` or equivalent. This is the main Presence seam for terminal participation. |
| FD 0/1/2 TTY awareness | FD table at spawn | Presence | Annotate or replace stdin/stdout/stderr with explicit terminal references once Presence exists. |

---

## 3. Migration status legend

| Status | Meaning |
|--------|---------|
| **bridged ✓** | A bridge module exists and is the canonical public surface; new code must go through the bridge. |
| bridge in place | Bridge exists but raw field still serves as the only backing. |
| extract next | Low-risk; extraction is the next logical step. Prerequisites are in place. |
| keep for now | No extraction until a prerequisite concept (Place, Presence, Group) is introduced. |
| quarantine | Unix baggage; isolated, not yet removable, do not deepen. |
| not yet added | Field does not exist; must be introduced before the associated concept can be extracted. |
| remove later | Will be deleted once extraction is complete and all callers updated. |

---

## 4. What is already bridged (Phase 8 baseline)

| Bridge module | What it surfaces | Phase |
|---|---|---|
| `kernel::place::bridge` | `cwd` + `namespace_label` → `thingos::place::Place` | 8 |
| `kernel::group::bridge` | `pgid` + `sid` + `session_leader` → `thingos::group::Group` | 4 |
| `kernel::authority::bridge` | `exec_path` → `thingos::authority::Authority::name` | 7 |

---

## 5. What is not yet bridged or extracted

| Responsibility | Blocker / note |
|---|---|
| Per-process namespace isolation | `NamespaceRef` is a unit struct; all processes share global mount table. Defer until namespace work. |
| Per-process chroot / filesystem root | No root field on `Process`; `Place::root` hardcoded to `"/"`. Must add field before extraction. |
| Inherited environment blob | No Place-facing env model; quarantined as Unix legacy. |
| Controlling TTY attachment | No Presence concept yet; implicit acquisition via `maybe_acquire_controlling_tty`. |
| `foreground_pgid` query outside devfs | Global TTY state locked inside `devfs`; `group::bridge` uses heuristic fallback. |
| Session lifecycle (setsid / SIGHUP) | Needs formal Group/Presence model before `setsid` can become explicit Group-create + Presence-detach. |
| FD 0/1/2 as explicit TTY Presence | No TTY-annotated FD concept; stdin/stdout/stderr TTY-ness is implicit. |
| Per-thread execution context (TLS, signal mask) | Extraction blocked on signal authority and Space/Task split. |

---

## 6. Constraints and notes for future issues

- **Do not conflate Place with just cwd.** Place owns the full world-context
  boundary: cwd, namespace view, and filesystem root.  All three must move
  together.
- **Do not omit TTY/session baggage.** `pgid`, `sid`, `session_leader`, and
  `ConsoleTtyState` are execution-context state and must be tracked here, even
  though they belong to Group/Presence rather than Place.
- **Do not treat Process as the long-term owner** of any of the items in this
  inventory.  All of them are transitional.
- **Presence is not yet introduced.** Terminal attachment (`controlling_sid`,
  `foreground_pgid`, implicit acquisition) belongs to Presence, not Place.
  New code must not conflate world-context (Place) with person-in-place
  (Presence).
- **Do not perform refactors in response to this document alone.** Each
  extraction item should be scoped in a dedicated follow-on issue that cites
  this inventory as the canonical planning reference.

---

## Related documents

- `docs/migration/process_responsibility_map.md` — canonical Process field decomposition and extraction order
- `docs/migration/authority_inventory.md` — credential/permission inventory (companion)
- `docs/concepts/process-object.md` — Process / Thread struct design
- `docs/concepts/namespaces.md` — namespace behaviour matrix and roadmap
- `docs/concepts/janix-guardrails.md` — architecture guardrails for all kernel changes
- `kernel/src/task/mod.rs` — primary `Process` struct with inline migration annotations
- `kernel/src/place/bridge.rs` — Phase 8 world-context bridge
- `kernel/src/group/bridge.rs` — Phase 4 coordination bridge
- `kernel/src/vfs/devfs.rs` — TTY/console device state (`ConsoleTtyState`)
- `thingos/src/place.rs` — canonical `Place` type (cwd, namespace, root)

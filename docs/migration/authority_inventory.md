# Authority Inventory: Process-Carried Credential and Permission State

> **Status**: Phase 9 baseline inventory — initial mapping artifact.
> This document is observational, not aspirational. Do not implement the future
> authority model yet; first understand the present one faithfully.
>
> Companion document to `docs/migration/process_responsibility_map.md`.

---

## Purpose

This document is the **structured migration artifact** for the issue
_"Inventory Process-Carried Credential and Permission State"_.

It inventories every field and code path that currently answers the question
"can this thing do that thing?" in Thing-OS, maps each item to its intended
future owner, and records the sequencing constraints that govern safe extraction.

Sources consulted:

| File | Coverage |
|------|----------|
| `kernel/src/task/mod.rs` | `Process`, `Thread<R>`, all subdivisions |
| `kernel/src/authority/bridge.rs` | Authority bridge and inventory table |
| `kernel/src/ipc/handles.rs` / `ipc/mod.rs` | Handle table, channel IPC |
| `kernel/src/syscall/dispatch.rs` | Full syscall surface |
| `kernel/src/syscall/handlers/device.rs` | Device claim / MMIO / IRQ / DMA |
| `kernel/src/syscall/handlers/process.rs` | Spawn, exec, kill, priority, reboot |
| `kernel/src/syscall/handlers/signal.rs` | kill, setpgid, setsid |
| `kernel/src/syscall/handlers/vfs.rs` | open, mount, chmod, all VFS ops |
| `kernel/src/vfs/fd_table.rs` | Per-process FD table |
| `docs/migration/process_responsibility_map.md` | Pre-existing migration notes |

---

## 1. Authority Inventory Table

All fields, tokens, and implicit state that influence "can this thing do that
thing?" are listed below. Items are grouped by their current location.

### 1.1 `Process` struct (`kernel/src/task/mod.rs`)

| Authority item | Field path | Description | Read by | Written by | Enforced? | Notes |
|---|---|---|---|---|---|---|
| Process identity | `Process.pid` | Thread-Group ID (TGID) | syscalls, scheduler, VFS, IPC | spawn/exec | implicit | Doubles as TGID and address-space ID; split deferred |
| Parent identity | `Process.lifecycle.ppid` | PID of the parent | `waitpid`, `getppid` | spawn | implicit | Only used for waitpid routing; no permission check |
| Thread membership | `Process.lifecycle.thread_ids` | List of TIDs in this group | scheduler, exec collapse | spawn_thread, thread exit | enforced (exec gate) | Used by exec_in_progress gate and group-exit logic |
| Exec gate | `Process.lifecycle.exec_in_progress` | Prevents new thread spawns during exec | `spawn_user_thread` | exec commit/rollback | enforced | Only lifecycle gate enforced today |
| Waitpid queue | `Process.lifecycle.children_done` | Exit statuses of dead children | `waitpid` | child exit | enforced indirectly | Consumes exit events; no auth check on who calls waitpid |
| FD table | `Process.fd_table` | Open file descriptors (handles to VFS nodes) | all VFS syscalls | open/dup/close/exec | enforced (possession) | Authority by possession; no per-FD capability model |
| Current directory | `Process.cwd` | Current working dir (affects path resolution) | VFS path resolver | `chdir`, `spawn_process_ex` (cwd override) | implicit | Affects all relative path operations; no isolation |
| VFS namespace | `Process.namespace` | Mount-table view | VFS path resolver | spawn (inherited) | not yet | Unit struct today; all processes share global mount table |
| Executable path | `Process.exec_path` | Path of running image | `authority::bridge` (name fallback) | exec | implicit | Used only as identity label, not for permission |
| Address space | `Process.space.aspace_raw` | Page-table root token | scheduler, memory syscalls | exec, spawn | enforced (hardware) | Enforced by MMU; not process-level policy |
| VM mappings | `Process.space.mappings` | Virtual memory map | scheduler, `vm_map`/`vm_protect` | exec, mmap, munmap | enforced (MMU) | Mapping metadata; real enforcement in page tables |
| Signal state | `Process.unix_compat.signals` | Per-process signal dispositions, pending set, stop/alarm state | signal delivery, `sigaction`, `sigprocmask` | `sigaction`, signal send | partial | No auth on who sends; SIGKILL/SIGSTOP not blockable; job-control mixed in |
| Process group ID | `Process.unix_compat.pgid` | Process group membership | `kill(0/neg)`, job-control, `getpgrp` | `setpgid`, `setsid`, spawn (inherited) | partial | Inherited at spawn; guards group-signal delivery loosely |
| Session ID | `Process.unix_compat.sid` | Session identity | job-control logic, TTY | `setsid` | partial | Session leader enforced only weakly |
| Session leader flag | `Process.unix_compat.session_leader` | TTY foreground ownership proxy | `group::bridge` | `setsid` | assumed | Used as heuristic; no formal TTY authority model |
| Environment map | `Process.unix_compat.env` | Inherited environment variables | `sys_env_get/set/list` | spawn (inherited), `sys_env_set` | not enforced | Any process can set/unset own env; affects PATH etc. |
| Argument vector | `Process.unix_compat.argv` | Spawn-time arguments | `sys_argv_get` | spawn/exec | read-only after spawn | No authority role; included for completeness |
| ELF aux vector | `Process.unix_compat.auxv` | ELF AT_* entries (stack layout, vDSO, uid/gid stubs) | `sys_auxv_get` | exec | read-only after exec | AT_UID/AT_GID fields are zeros (no uid/gid yet) |
| Message inbox | `Process.unix_compat.message_inbox` | Typed message queue | `SYS_MSG_RECV` (future) | message delivery | partial (bounded) | Bounded capacity; sender TID recorded in metadata but not verified |

### 1.2 `Thread<R>` struct (`kernel/src/task/mod.rs`)

| Authority item | Field path | Description | Read by | Written by | Enforced? | Notes |
|---|---|---|---|---|---|---|
| Thread identity | `Thread.id` (ThreadId) | Unique schedulable-entity ID | scheduler, device registry, signal delivery | spawn | implicit | Used in claim ownership (device), signal routing |
| Kernel/user mode | `Thread.is_user` | Distinguishes kernel threads from user threads | scheduler, VFS syscalls | spawn | enforced (hardware) | Kernel threads cannot invoke user VFS syscalls; implicit trust distinction |
| Process reference | `Thread.process_info` | Back-reference to owning `Process` | all process-facing syscalls | spawn | enforced (None → ENOENT) | Kernel threads have `None`; VFS ops check for presence |
| Signal mask | `Thread.signals` (ThreadSignals) | Per-thread signal mask + thread-directed pending signals | signal delivery | `sigprocmask` | partial | Masks signals; no authority check on mask changes |
| Exit code | `Thread.exit_code` | Thread exit state | `task_wait`, `waitpid` via job bridge | `exit` | implicit | Not an authority field but governs reaping |
| Priority | `Thread.priority` | Scheduling priority | scheduler | `set_priority` | not enforced | **Any** thread can raise its own or another thread's priority — no cap |
| TLS base | `Thread.user_fs_base` | User-mode FS base (TLS pointer) | user code, syscall return path | `SYS_TASK_SET_TLS_BASE` | not checked | Userspace can point TLS anywhere; kernel does not validate |
| SIMD state | `Thread.simd` | Saved SIMD/FPU registers | context switch | context switch | N/A | Not authority-relevant |

### 1.3 IPC / Handle Table (`kernel/src/ipc/`)

| Authority item | Location | Description | Read by | Written by | Enforced? | Notes |
|---|---|---|---|---|---|---|
| Global handle table | `ipc::GLOBAL_HANDLE_TABLE` (static `Mutex<HandleTable>`) | Single system-wide mapping from handle number → port + mode | all channel syscalls | `channel_create`, `recv_handle` | enforced (mode check) | **Single-process model (v0)**: no per-process isolation; any thread knowing a handle number can use it. `HandleMode::Read`/`Write` enforced by `table.get(h, mode)`. |
| Channel read handle | `HandleTable` entry, `mode=Read` | Right to receive messages from a port | `SYS_CHANNEL_RECV`, `SYS_CHANNEL_RECV_MSG` | `channel_create`, `recv_handle` | enforced (mode) | Enforced at get; no ownership transfer check |
| Channel write handle | `HandleTable` entry, `mode=Write` | Right to send messages to a port | `SYS_CHANNEL_SEND`, `SYS_CHANNEL_SEND_MSG` | `channel_create`, `recv_handle` | enforced (mode) | Enforced at get; no transfer audit trail |
| Handle-in-message | `SYS_CHANNEL_SEND_MSG` / `SYS_CHANNEL_RECV_MSG` | A handle number embedded in a channel message payload | receiving process | sender | not enforced | Receiver obtains the raw handle number; kernel does not check whether the sender legitimately owns that handle before embedding it |

### 1.4 Device Registry (`kernel/src/device_registry/`, `kernel/src/syscall/handlers/device.rs`)

| Authority item | Location | Description | Read by | Written by | Enforced? | Notes |
|---|---|---|---|---|---|---|
| Device claim handle | `device_registry::REGISTRY`, claim slot | Exclusive ownership token for a PCI device | `sys_device_map_mmio`, `sys_device_irq_subscribe`, `sys_device_irq_wait`, `sys_device_alloc_dma` | `sys_device_claim` | enforced (TID check) | `verify_claim(handle, task_id)` enforces per-TID ownership of MMIO, IRQ, DMA. First-claimer wins; re-claim returns `EBUSY`. |
| I/O port access | `sys_device_ioport` | Raw x86 I/O port read/write | userspace driver | userspace driver | **not enforced** | **No claim check on `SYS_DEVICE_IOPORT_READ/WRITE`**. Any process can access any I/O port. |
| IRQ vector subscription | `sys_device_irq_subscribe` (VECTOR mode) | Subscribe to a raw CPU interrupt vector | any process | any process | **not enforced** | Claim check only applied in `DEVICE_IRQ_SUBSCRIBE_DEVICE` mode. `DEVICE_IRQ_SUBSCRIBE_VECTOR` fallthrough skips claim verification. |
| DMA physical address | `sys_device_dma_phys` | Translate user-VA to physical address | DMA user driver | DMA allocation path | partial | Any caller can query physical address of its own VA; no check that VA is a DMA region |

### 1.5 VFS Layer (`kernel/src/vfs/`, `kernel/src/syscall/handlers/vfs.rs`)

| Authority item | Location | Description | Read by | Written by | Enforced? | Notes |
|---|---|---|---|---|---|---|
| Open file node | `FdTable` entry | VFS node + flags (O_RDONLY, O_WRONLY, O_RDWR) | read/write/stat/poll syscalls | `open`, `dup`, `dup2` | enforced (flag check) | `OpenFlags` read/write bits enforced at syscall boundary |
| FD close-on-exec flag | `FdTable::OpenFile.fd_flags` | Whether to close FD at exec | exec commit | `fcntl(FD_CLOEXEC)` | enforced | Applied during exec; does not gate initial open |
| File offset | `FdTable::OpenFile.offset` (shared `Arc<Mutex<u64>>`) | Current read/write position | read/write/seek | read/write/seek/dup | shared across dups | Shared on dup/dup2; authority-neutral (position only) |
| Permission bits (`mode`) | VFS node metadata | POSIX-style `rwxrwxrwx` + setuid/setgid/sticky | `stat` | `chmod`, `fchmod` | **not enforced** | Bits stored but **never checked on open/read/write**. `chmod`/`fchmod` accept any mode from any caller without capability check. |
| Mount point | `vfs::mount::MOUNTS` | Controls which filesystem serves a path prefix | all path-resolution calls | `sys_fs_mount`, `sys_fs_umount` | **not enforced** | **Any process can mount or unmount any path.** No privilege check on `SYS_FS_MOUNT` or `SYS_FS_UMOUNT`. |

### 1.6 Implicit / Ghost Authority

| Authority item | Location | Description | Enforcement | Notes / Risk |
|---|---|---|---|---|
| Process kill | `sys_task_kill` | Any thread can kill any other thread by TID | none | `kill_by_tid_current` does not check caller ownership or relationship |
| Arbitrary signal send | `sys_kill` | Any process can send any signal to any other PID or group (except broadcast `pid=-1`) | none (ESRCH only) | No sender/receiver relationship check; SIGKILL is thus available to all |
| System reboot/halt | `sys_reboot` | Any process can reboot or halt the entire system | none | No privilege check whatsoever |
| Log level set | `dispatch.rs::SYS_LOG_SET_LEVEL` | Any process can change the global kernel log level | none | Inline in dispatch; no authority check |
| Console disable | `sys_console_disable` | Any process can disable the system console | none | No privilege check |
| Thread priority escalation | `sys_set_priority` | Any thread can set any other thread to `Realtime` priority | none | Can starve all other threads |
| Spawn executable | `sys_spawn_process_ex` | Any process can spawn any VFS-visible executable with any argv/env/cwd | file existence only | Authority flows from VFS open (which has no access control) |
| Exec self | `sys_task_exec` | Any process can exec itself with any VFS executable and env | file existence only | No privilege check beyond being able to open the file |

---

## 2. Permission Check Index

This section catalogs all locations where a permission decision is currently
made. Where a check is absent but should exist, it is noted as **missing**.

### 2.1 Syscall handlers (`kernel/src/syscall/handlers/`)

| Syscall | File | What is checked | Data used | Check type | Complete? | Notes |
|---|---|---|---|---|---|---|
| `SYS_SPAWN_THREAD` | `process.rs` | exec_in_progress gate | `Process.lifecycle.exec_in_progress` | lifecycle | yes (in scope) | Only prevents concurrent thread+exec |
| `SYS_SPAWN_THREAD` | `process.rs` | Stack layout sanity | request fields | range validation | yes | Not authority, safety |
| `SYS_TASK_KILL` | `process.rs` | Target TID existence | scheduler | existence only | **missing** | No relationship check; any thread can kill any other |
| `SYS_SET_PRIORITY` | `process.rs` | Priority range [0–4] | request arg | bounds only | partial | No cap on Realtime escalation |
| `SYS_KILL` | `signal.rs` | Signal number range | sig arg | bounds | yes | No sender/receiver permission check |
| `SYS_SETPGID` | `signal.rs` | `setpgid_current` internal logic | pgid, current session | partial POSIX rules | partial | Incomplete POSIX enforcement; no uid check |
| `SYS_SETSID` | `signal.rs` | Not already a group leader | `pgid == pid` check | existence | partial | No extra privilege required |
| `SYS_REBOOT` | `process.rs` | Command constant valid | cmd arg | bounds only | **missing** | No privilege check |
| `SYS_LOG_SET_LEVEL` | `dispatch.rs` | None | none | — | **missing** | Zero-gate privileged operation |
| `SYS_CONSOLE_DISABLE` | `handlers/mod.rs` | None (assumed) | none | — | **missing** | No privilege check confirmed |
| `SYS_FS_OPEN` | `vfs.rs` | Process has process_info | scheduler hook | existence | partial | No path/mode access control |
| `SYS_FS_CHMOD` / `SYS_FS_FCHMOD` | `vfs.rs` | None (delegates to node) | node.chmod | driver-dependent | partial | Caller identity not checked; depends on node implementation |
| `SYS_FS_MOUNT` | `vfs.rs` | None | path string | — | **missing** | Any process can mount at any path |
| `SYS_FS_UMOUNT` | `vfs.rs` | None | path string | — | **missing** | Any process can unmount any mount point |
| `SYS_DEVICE_CLAIM` | `device.rs` | Device slot exists; not already claimed | device registry + TID | ownership (first-claimer) | enforced | Single-claim exclusive ownership |
| `SYS_DEVICE_MAP_MMIO` | `device.rs` | `verify_claim(handle, task_id)` | device registry + TID | ownership | enforced | TID-bound claim check |
| `SYS_DEVICE_IRQ_SUBSCRIBE` (DEVICE mode) | `device.rs` | `verify_claim(handle, task_id)` | device registry + TID | ownership | enforced | Claim check present |
| `SYS_DEVICE_IRQ_SUBSCRIBE` (VECTOR mode) | `device.rs` | None | none | — | **missing** | Fallthrough skips claim check; any process |
| `SYS_DEVICE_IOPORT_READ/WRITE` | `device.rs` | None | none | — | **missing** | Unrestricted I/O port access |
| `SYS_DEVICE_ALLOC_DMA` | `device.rs` | `verify_claim(handle, task_id)` | device registry + TID | ownership | enforced | Claim check present |
| `SYS_CHANNEL_SEND/RECV` | `port.rs` | Handle valid + mode match | `GLOBAL_HANDLE_TABLE` | possession + mode | partial | Global table; no per-process ownership isolation |
| `SYS_CHANNEL_SEND_MSG` | `port.rs` | Handle valid + mode; embedded handles not verified | handle table | possession only | partial | Sender can embed arbitrary handle numbers |
| `SYS_VM_MAP` / `SYS_VM_UNMAP` | `memory.rs` | Address range in user space | address validator | range | partial | No capability for mapping device/special memory |
| `SYS_VM_PROTECT` | `memory.rs` | Address range in user space | address validator | range | partial | No escalation check (can mark exec) |
| `SYS_ENTROPY_SEED` | `random.rs` | (need to verify) | — | — | likely **missing** | Seeding system entropy; should be privileged |
| `SYS_GETRANDOM` | `random.rs` | None needed | — | N/A | N/A | Reading entropy is not privileged |

### 2.2 VFS node access control

| Operation | File | Check | Complete? | Notes |
|---|---|---|---|---|
| `VfsNode::read` | driver-specific | None at VFS layer | **missing** | Permission bits on node never gated |
| `VfsNode::write` | driver-specific | None at VFS layer | **missing** | Write access not checked against mode bits |
| `VfsNode::chmod` | driver-specific | Depends on driver | partial | ramfs supports it; devfs may not; no caller check |
| Mount point lookup | `vfs::mount` | Path prefix matching only | implicit | No namespace isolation; all processes share mounts |

### 2.3 Device access paths

| Device | Access check | Notes |
|---|---|---|
| `/dev/fb0` (framebuffer) | open succeeds if node exists; debug logs only | Any process that can open `/dev/fb0` owns the framebuffer |
| PCI devices | exclusive claim by TID required before MMIO/IRQ/DMA | Enforced |
| I/O ports | **none** | Any process can issue `IN`/`OUT` instructions via `SYS_DEVICE_IOPORT_*` |
| `/dev/urandom` (random) | open only | Read-only; no write |
| Pipes / channels | possession of read or write handle | Mode enforced; no ownership audit |

---

## 3. Authority Flow Map

This section describes how authority moves through the system at key lifecycle
events.

### 3.1 Process / task creation (`SYS_SPAWN_PROCESS_EX`)

```
Parent process
  │
  ├─ cwd (Option<String>)           → child Process.cwd  (parent's cwd if None)
  ├─ env (BTreeMap)                 → child Process.unix_compat.env (full copy)
  ├─ pgid / sid                     → child inherits via ProcessUnixCompat::inherit()
  ├─ fd_remap (up to 64 entries)    → mapped into child Process.fd_table (0,1,2 = stdio)
  ├─ inherited_handles (up to 8)   → handle numbers copied into child context
  │                                   (no ownership transfer audit)
  └─ exec_path                      → child Process.exec_path (authority name label)

Child process receives:
  - new pid (allocated by scheduler)
  - ppid = parent's pid
  - pgid = parent's pgid (unless overridden by setpgid)
  - sid = parent's sid
  - empty signal dispositions (ProcessSignals::new())
  - address space: freshly loaded ELF image
  - NO capability set (empty)
  - NO uid/gid (not yet implemented)
```

**Authority widening risk**: The caller does not need any capability to spawn a
process. The child inherits the caller's cwd, env, and pgid without restrictions.

### 3.2 exec (`SYS_TASK_EXEC`)

```
Current process
  │
  ├─ fd_table: close_on_exec() applied (FD_CLOEXEC FDs dropped)
  ├─ address space: replaced with new ELF image
  ├─ exec_path: updated to new binary
  ├─ signals: reset to default dispositions (ProcessSignals::new())
  ├─ argv: replaced from request
  ├─ env: replaced from request  ← no filtering; caller controls full env
  └─ cwd: unchanged across exec  ← carries forward implicitly
```

**Inheritance invariant**: cwd persists across exec. FDs without `FD_CLOEXEC`
persist. There is no mechanism to force-drop authority on exec.

### 3.3 IPC handle passing (`SYS_CHANNEL_SEND_MSG`)

```
Sender
  │
  ├─ encodes handle numbers as u32 values in the message payload
  └─ sends message to GLOBAL_HANDLE_TABLE entry (port)

Receiver
  ├─ reads data + handle[] from channel
  └─ uses raw handle numbers directly against GLOBAL_HANDLE_TABLE
     (no transfer record; kernel does not verify sender owned the handles)
```

**Risk**: Handle numbers are global in v0. A sender can embed handle numbers
it does not legitimately own. The receiver trusts them unconditionally.

### 3.4 Device access

```
Userspace driver
  │
  ├─ SYS_DEVICE_CLAIM(slot) → claim_handle (u32) bound to calling TID
  │                            ↳ REGISTRY.verify_claim(handle, task_id) enforced on:
  │                               • SYS_DEVICE_MAP_MMIO
  │                               • SYS_DEVICE_IRQ_SUBSCRIBE (DEVICE mode)
  │                               • SYS_DEVICE_ALLOC_DMA
  │
  └─ SYS_DEVICE_IOPORT_READ/WRITE → unchecked; raw hardware access
```

**Implicit trust**: Any process that calls `SYS_DEVICE_IOPORT_*` can access
arbitrary hardware I/O ports without a prior claim.

### 3.5 Signal delivery

```
Sender: any process with a known PID/PGID
  │
  ├─ SYS_KILL(pid>0, sig) → process exists? deliver → no sender check
  ├─ SYS_KILL(pid==0, sig) → all in sender's pgid → inherits group membership
  └─ SYS_KILL(pid<-1, sig) → all in group (-pid) → no group-membership check

Receiver: Process.unix_compat.signals
  ├─ disposition checked (SIG_DFL / SIG_IGN / handler)
  ├─ signal mask checked per-thread (Thread.signals)
  └─ SIGKILL / SIGSTOP: never blockable, never ignorable
```

**Missing check**: The sender is never checked for permission to signal the
target. POSIX requires uid match or `CAP_KILL`; neither is implemented.

### 3.6 VFS path resolution

```
Process
  ├─ cwd (absolute path string) + namespace (global)
  │
  └─ path → abs_path via resolve_abs_path(cwd, path)
             ↓
             vfs::mount::lookup(abs_path)
             ↓
             VfsNode (no mode check, no caller credential check)
             ↓
             inserted into fd_table (OpenFlags mode bits only)
```

**No access control**: Path resolution does not check ownership, mode bits,
or caller credentials at any step. Any process that knows a path can open it.

---

## 4. Intended Ownership Mapping

For each authority item, the conceptual future home in the ThingOS object model.

| Authority item | Current location | Proposed future home | Notes |
|---|---|---|---|
| Process identity (pid) | `Process.pid` | `Job` (lifecycle ID) + `Space` (VM ID) | Requires Job/Space split |
| Parent/child linkage | `ProcessLifecycle` | `Job` | Subdivision already in place |
| Thread group membership | `ProcessLifecycle` | `Job` | Subdivision already in place |
| Exec gate | `ProcessLifecycle` | `Job` | Lives with lifecycle |
| FD table | `Process.fd_table` | Handle table (Phase 9+) | No handle-table concept yet |
| cwd | `Process.cwd` | `Place` | Bridge exists; promote backing |
| VFS namespace | `Process.namespace` | `Place` | Per-process isolation deferred |
| Signal dispositions | `ProcessUnixCompat.signals` | `Authority` (dispositions) + `Group` (job-control) | Requires Authority stabilisation |
| Signal mask | `Thread.signals` | `Authority` (thread-scoped) | Complex; deferred |
| pgid / sid | `ProcessUnixCompat` | `Group` | Bridges exist |
| Session leader flag | `ProcessUnixCompat` | `Group` / `Presence` | Presence not yet introduced |
| env map | `ProcessUnixCompat` | `Place` or `Authority` context | No clean home yet; keep quarantined |
| argv / auxv | `ProcessUnixCompat` | Spawn record (Job) | Introduce spawn record first |
| Channel handle table | `GLOBAL_HANDLE_TABLE` (global) | Per-process handle table with ownership | Requires process-scoped handle tables |
| Handle-in-message | message payload | Capability transfer with sender verification | Capability object system required |
| Device claim | device registry + TID | Capability tied to handle / Authority object | Extend from current claim model |
| I/O port access | unchecked | Capability / device claim (should gate ioport) | Straightforward addition to claim model |
| IRQ vector subscription | unchecked (VECTOR mode) | Device claim (unify with DEVICE mode) | Low-effort fix |
| File permission bits | VFS node stat.mode | Enforced at VFS open layer against caller Authority | Requires caller credential in VFS |
| Mount authority | unchecked | Authority capability (`CAP_MOUNT`-equivalent) | Requires Authority in VFS layer |
| Process kill (by TID) | unchecked | Job-scoped kill or Authority capability | Requires relationship model |
| Signal send | unchecked | Authority capability or job membership | Requires uid/gid or capability |
| Reboot/halt | unchecked | Authority capability (`CAP_REBOOT`-equivalent) | Single line to gate |
| Log level set | unchecked | Authority capability | Trivial to gate once Authority is in syscall path |
| Console disable | unchecked | Authority capability | Same as log level |
| Priority escalation | unchecked | Authority capability (`CAP_SYS_NICE`-equivalent) | Gate at `sys_set_priority` |
| uid/gid | **not present** | `Authority` fields | Must be added to Process first |
| Capability mask | **not present** | `Authority::capabilities` | Must be added to Process first |
| Service-account / principal | **not present** | `Authority` | Future identity concept |

---

## 5. Extraction Sequencing Notes

Based on the inventory, the following sequencing reduces cascading breakage.
Phases align with existing `docs/migration/process_responsibility_map.md`.

### 5.1 Low-risk extraction candidates

These items are already well-bounded and can be moved with minimal risk:

| Priority | Item | Why low-risk |
|---|---|---|
| 1 | Gate `SYS_DEVICE_IOPORT_*` behind device claim | Straightforward check; precedent in MMIO handler |
| 2 | Unify `SYS_DEVICE_IRQ_SUBSCRIBE` VECTOR mode under claim check | One-line fix; same pattern as DEVICE mode |
| 3 | Gate `SYS_REBOOT` / `SYS_LOG_SET_LEVEL` / `SYS_CONSOLE_DISABLE` | Add a privileged-caller flag; trivial once first Authority concept in syscall path |
| 4 | Per-process handle table (replace `GLOBAL_HANDLE_TABLE`) | Comment in code says "Single Process Model v0"; this is the designated migration point |

### 5.2 Medium-risk items (require prerequisite concepts)

| Priority | Item | Prerequisite |
|---|---|---|
| 5 | Enforce VFS permission bits at open | Authority in syscall path; uid/gid or capability field in Process |
| 6 | Gate `SYS_FS_MOUNT` / `SYS_FS_UMOUNT` | Authority capability concept |
| 7 | Gate `SYS_KILL` by sender/receiver relationship | uid/gid or job membership model |
| 8 | Gate `SYS_TASK_KILL` by job/group relationship | Job kernel object |
| 9 | Gate `SYS_SET_PRIORITY` Realtime escalation | Authority capability |
| 10 | Handle-in-message sender verification | Per-process handle tables + ownership transfer protocol |

### 5.3 High-risk tightly-coupled areas

| Item | Coupling / Risk |
|---|---|
| Signal state (`ProcessSignals` / `ThreadSignals`) | Split between Authority (dispositions) and Group (job-control) is complex; signal delivery is on the hot path |
| `GLOBAL_HANDLE_TABLE` → per-process tables | All IPC users assume global handles; migration touches every channel caller |
| VFS permission bit enforcement | Currently zero callers check mode bits; enabling checks will break all existing processes until they get credentials |
| uid/gid introduction | Requires adding fields to `Process`, updating `authority::bridge`, updating all spawn/exec paths, and deciding on default values |

### 5.4 Prerequisite refactors

Before attempting high-risk items, the following must exist:

1. **`Job` kernel object** — promote `ProcessLifecycle` into a first-class `Job`.  
   Needed for: kill permission by relationship, waitpid scoping, session authority.

2. **uid/gid fields in `Process`** — add to `Process` (or Authority substructure),
   surface through `authority::bridge`.  
   Needed for: signal permission, VFS access control, priority gating.

3. **Authority in syscall path** — make Authority derivable at syscall entry
   (not just via procfs bridge) so handlers can check capabilities.  
   Needed for: all capability-gated syscalls.

4. **Per-process handle tables** — replace `GLOBAL_HANDLE_TABLE` with
   per-process tables.  
   Needed for: secure handle-in-message transfer, handle ownership.

5. **`Space` kernel object** — separate address-space identity from lifecycle identity.  
   Needed for: clean `pid`/TGID separation; multi-threaded space sharing.

### 5.5 Natural phase boundaries

| Phase | Description |
|---|---|
| Phase 10 | Introduce `Job`; promote `ProcessLifecycle`; move exit/wait semantics |
| Phase 11 | Add uid/gid to `Process`; surface through `authority::bridge` |
| Phase 12 | Authority in syscall path; gate reboot/log/console/ioport |
| Phase 13 | Per-process handle tables; gate handle-in-message transfer |
| Phase 14 | VFS access control (enforce mode bits, gate mount) |
| Phase 15 | Signal permission model (uid check or capability) |

---

## Ambiguity and Risk Register

Items marked as requiring follow-up investigation or design clarification:

| Item | Risk | Classification |
|---|---|---|
| `GLOBAL_HANDLE_TABLE` single-process model | Any thread can use any handle if it guesses the index | **assumed invariant** (from v0 design) — needs explicit removal |
| `SYS_DEVICE_IOPORT_*` unchecked | Arbitrary hardware access from any process | **likely bug** — easy to fix with claim check |
| `SYS_DEVICE_IRQ_SUBSCRIBE` VECTOR mode | Unchecked vector subscription | **likely oversight** — one fallthrough clause |
| `SYS_KILL` no sender check | Arbitrary cross-process signaling | **assumed invariant** — intentional for now, must be addressed with uid/gid |
| `SYS_TASK_KILL` no relationship check | Any thread can kill any TID | **assumed invariant** — dangerous in multi-process scenarios |
| `SYS_REBOOT` unchecked | Any process can halt the system | **assumed invariant** (single-user embedded design) — easy to gate |
| Signal in-message sender metadata | `sender_tid` is recorded but never verified | **assumed invariant** — not validated at delivery |
| `chmod`/`fchmod` without owner check | Any process can change mode of any file | **assumed invariant** — safe only in single-user model |
| `env` as authority vector | `PATH` and other env vars inherited; attacker-controlled env could redirect exec | **unclear ownership** — env not classified as authority today |
| AT_UID / AT_GID in auxv are zeros | Programs expecting uid/gid from auxv see 0 (root equivalent) | **assumed invariant** — will cause issues when uid/gid introduced |
| `cwd` not sanitised across exec | Path traversal possible if exec replaces binary with different trust level | **assumed invariant** |
| `namespace` unit-struct global | All processes share mount namespace; any process can shadow paths | **duplicate enforcement gap** — per-process namespaces deferred |

---

## Related Documents

- `docs/migration/process_responsibility_map.md` — full `Process` field migration map
- `kernel/src/authority/bridge.rs` — Phase 7 permission bridge (current source of truth for authority extraction status)
- `kernel/src/task/mod.rs` — primary `Process` and `Thread<R>` structs with inline migration annotations
- `kernel/src/ipc/handles.rs` — handle table comment: "Per-process capability-gated port access" (aspirational; currently global)
- `kernel/src/syscall/dispatch.rs` — complete syscall surface
- `docs/concepts/janix-guardrails.md` — architecture guardrails for all kernel changes

# Janix Architecture Guardrails

> **Non-negotiable design principles for the janix kernel and its userland.**
> These rules exist to prevent architectural regressions during large refactors.
> Every PR that touches `kernel/`, `abi/`, `bran/`, or `userspace/` must be
> evaluated against this checklist.

---

## 1. Scheduler-First Execution Model

**Rule:** The kernel scheduler is the sole authority that grants CPU time.
Every unit of work must be a scheduled *task*.  Nothing may execute outside
of a task context.

### What this means in practice

- All execution — kernel threads, drivers, user processes — is represented by
  a `TaskId`.
- The scheduler (in `kernel/src/sched/`) owns the run-queues and is the only
  component that transitions tasks between `Runnable`, `Running`, and `Blocked`
  states.
- Priority aging (see `docs/scheduler-anti-starvation.md`) prevents starvation
  of low-priority tasks.
- Timer interrupts drive preemption; no code path may hold a spinlock across a
  reschedule point unless it is prepared for the resulting priority inversion.

### What is NOT allowed

- Busy-polling loops that never yield (use `SYS_YIELD` or sleep primitives).
- Ad-hoc "run immediately" invocations that bypass the scheduler.
- Calling `yield_now` while holding the global scheduler lock.

### Tests / Validation

- `docs/scheduler-anti-starvation.md` describes the aging invariants.
- Scheduler unit tests live in `kernel/src/sched/`.

---

## 2. Userland-Driver Direction

**Rule:** Hardware drivers live in userspace.  The kernel provides only the
minimal primitives needed to grant controlled access: MMIO mapping, DMA
allocation, I/O-port access, and IRQ subscription.

### What this means in practice

- A userspace driver claims a device with `SYS_DEVICE_CLAIM` (`0x5000`) and
  is handed MMIO regions via `SYS_DEVICE_MAP_MMIO` (`0x5002`).
- The kernel never touches device registers directly (except during very early
  boot before the driver process is running).
- Driver presence is published through VFS: a NIC driver mounts its state at
  `/dev/net/<name>/`, a display driver at `/dev/fb<n>/`, etc.
- Drivers communicate with the rest of the system through ordinary file I/O
  and channels, not through in-kernel callbacks or shared global state.

### Kernel-level device primitives (all in `0x5000`–`0x5008`)

| Syscall | Number | Purpose |
|---------|--------|---------|
| `SYS_DEVICE_CLAIM` | `0x5000` | Exclusive ownership of a device |
| `SYS_DEVICE_CALL` | `0x5001` | Driver-defined ioctl equivalent |
| `SYS_DEVICE_MAP_MMIO` | `0x5002` | Map device MMIO into process address space |
| `SYS_DEVICE_ALLOC_DMA` | `0x5003` | Allocate physically contiguous DMA buffer |
| `SYS_DEVICE_DMA_PHYS` | `0x5004` | Translate DMA buffer to physical address |
| `SYS_DEVICE_IOPORT_READ` | `0x5005` | Read from I/O port |
| `SYS_DEVICE_IOPORT_WRITE` | `0x5006` | Write to I/O port |
| `SYS_DEVICE_IRQ_SUBSCRIBE` | `0x5007` | Subscribe to a hardware interrupt |
| `SYS_DEVICE_IRQ_WAIT` | `0x5008` | Block until subscribed interrupt fires |

### What is NOT allowed

- Adding new in-kernel driver logic for anything that can run in userspace.
- Polling device registers inside the kernel scheduler loop.
- Accessing device state through `ThingId` / graph nodes — use VFS paths.

---

## 3. VFS-First System Surface

**Rule:** Every system resource is exposed through a mounted filesystem path.
If something has state, it has a path.  If something has behavior, it has a
file.

### What this means in practice

- The kernel boot framebuffer is `/dev/fb0`.
- Network interfaces are `/dev/net/<name>/`.
- Process metadata is `/proc/<pid>/`.
- Ephemeral runtime state lives under `/run/`.
- Service presence is declared under `/services/`.
- Session/desktop configuration lives at `/session/desktop/`.

### Boot invariant

Bloom (the compositor) and any other early-boot service **must** reach first
paint using only VFS and file-descriptor state.  They must NOT depend on:

- `ThingId` lookups or graph node traversal.
- `UI_CROWN` or any graph-discovery mechanism.
- Kernel-internal state that is not exposed through a mounted path.

### Syscall surface for VFS (`0x4000`–`0x4024`)

The canonical system surface is the `SYS_FS_*` family.  Do not invent new
kernel interfaces to solve problems that can be solved by mounting a new
filesystem subtree.

### What is NOT allowed

- Services that can only be reached through in-kernel data structures.
- Adding non-VFS boot dependencies for display, networking, or audio bring-up.
- Using `ThingId` / graph edges as the primary mechanism for inter-service
  communication in new code.

---

## 4. Spawn + Exec as the Process Model

**Rule:** New processes are created with **`spawn` + `exec`**, not `fork`.
There is no `SYS_FORK` in janix; this is intentional and permanent.

### What this means in practice

- `SYS_SPAWN_PROCESS` (`0x1005`) or `SYS_SPAWN_PROCESS_EX` (`0x1006`) creates
  a new, empty process with an inherited (or explicit) VFS namespace.
- `SYS_TASK_EXEC` (`0x100D`) replaces the image of the calling task with a new
  executable, à la POSIX `execve`.
- The combination of `spawn` + `exec` is the standard and complete process
  launch sequence.  Shell-like `posix_spawn` semantics are emulated at the
  userspace level (in `stem` or a future libc shim), not in the kernel.

### Why no fork?

`fork` requires duplicating the entire virtual address space, including all
kernel-internal state.  In a system where drivers are userspace processes and
shared VFS namespaces are explicitly managed, `fork` creates ambiguous
ownership of open file descriptors, channels, and MMIO mappings.  The
`spawn` + `exec` model is safer, more auditable, and maps cleanly to
capability-based security.

### Relevant syscalls

| Syscall | Number | Purpose |
|---------|--------|---------|
| `SYS_SPAWN_THREAD` | `0x1004` | Spawn a new thread in the current process |
| `SYS_SPAWN_PROCESS` | `0x1005` | Spawn a new empty process |
| `SYS_SPAWN_PROCESS_EX` | `0x1006` | Spawn a process with explicit configuration |
| `SYS_TASK_EXEC` | `0x100D` | Replace current image (exec) |
| `SYS_TASK_WAIT` | `0x1007` | Wait for a task to exit |
| `SYS_WAITPID` | `0x1011` | POSIX-compatible wait |

### What is NOT allowed

- Adding `SYS_FORK` or any copy-on-write address-space duplication to the
  kernel.
- Treating "fork without exec" as a lightweight IPC mechanism.
- Spawning processes via any path that bypasses `SYS_SPAWN_PROCESS[_EX]`.

---

## Guardrail Summary

| Principle | Short form | Primary enforcement point |
|-----------|------------|--------------------------|
| Scheduler-first | Every execution unit is a scheduled task | `kernel/src/sched/` |
| Userland drivers | Hardware logic lives outside the kernel | `SYS_DEVICE_*` boundary |
| VFS-first | All resources reachable via filesystem paths | `SYS_FS_*` syscalls |
| Spawn + exec | No fork; new processes use spawn + exec | `SYS_SPAWN_*` + `SYS_TASK_EXEC` |

---

## PR Review Checklist

Use this when reviewing any change that touches `kernel/`, `abi/`, `bran/`,
`stem/`, or `userspace/`:

- [ ] Does the change add execution that is not managed by the scheduler?
- [ ] Does the change add device logic inside the kernel that could live in
      userspace?
- [ ] Does the change require reaching resources through graph nodes / `ThingId`
      instead of VFS paths?
- [ ] Does the change introduce `fork`-like semantics or duplicate a process
      address space?
- [ ] Does the change add boot-time dependencies on `UI_CROWN`, graph
      discovery, or `ThingId` lookups?

If any answer is **yes**, the change needs justification or rework before merge.

---

## See Also

- `docs/concepts/platform.md` — stem-as-std and no-`std` platform contract
- `docs/concepts/scheduling.md` — scheduler internals
- `docs/concepts/userland.md` — userspace threading model
- `docs/concepts/syscalls.md` — full syscall ABI reference
- `docs/concepts/namespaces.md` — namespace semantics and isolation roadmap
- `docs/scheduler-anti-starvation.md` — priority aging details
- `AGENTS.md` — quick orientation for automated agents
- `.github/PULL_REQUEST_TEMPLATE.md` — machine-readable PR checklist

# Signal Subsystem

ThingOS implements POSIX-compatible signal semantics for process control,
asynchronous notifications, and job control.  This document describes the
current kernel implementation, the userspace ABI, and known intentional gaps.

---

## Overview

Signals are asynchronous notifications delivered to a process or thread.  The
kernel checks for pending signals at every syscall return point and delivers
them before handing control back to userspace.

### Signal Numbers

Standard non-realtime signals (numbers 1–31) are implemented:

| Number | Name     | Default Action       |
|--------|----------|----------------------|
| 1      | SIGHUP   | Terminate            |
| 2      | SIGINT   | Terminate            |
| 3      | SIGQUIT  | Terminate (core)     |
| 4      | SIGILL   | Terminate (core)     |
| 5      | SIGTRAP  | Terminate (core)     |
| 6      | SIGABRT  | Terminate (core)     |
| 7      | SIGBUS   | Terminate (core)     |
| 8      | SIGFPE   | Terminate (core)     |
| 9      | SIGKILL  | **Terminate (uncatchable)** |
| 10     | SIGUSR1  | Terminate            |
| 11     | SIGSEGV  | Terminate (core)     |
| 12     | SIGUSR2  | Terminate            |
| 13     | SIGPIPE  | Terminate            |
| 14     | SIGALRM  | Terminate            |
| 15     | SIGTERM  | Terminate            |
| 17     | SIGCHLD  | Ignore               |
| 18     | SIGCONT  | Continue             |
| 19     | SIGSTOP  | **Stop (uncatchable)** |
| 20     | SIGTSTP  | Stop                 |
| 21     | SIGTTIN  | Stop                 |
| 22     | SIGTTOU  | Stop                 |

Realtime signals (SIGRTMIN = 32 … NSIG − 1 = 63) are reserved in the bitmask
but no queued delivery is implemented yet.

---

## Architecture

### Data Structures

```
crate::task::Thread              (per-thread, in the task registry)
  └── signals: ThreadSignals
        ├── mask: SigSet          — blocked signals (per-thread)
        └── pending: SigSet       — thread-directed pending signals

crate::task::Process             (per-process, shared by thread group)
  └── signals: ProcessSignals
        ├── actions: [SigAction; 63]  — per-signal dispositions
        ├── pending: SigSet           — process-directed pending signals
        └── alarm_deadline: u64       — SIGALRM tick deadline
  └── children_done: VecDeque<(u32, i32)>  — reaped child statuses
```

### Delivery Path

1. Every syscall eventually calls `kernel_dispatch_flat` (in `kernel/src/syscall/flat.rs`).
2. On return from dispatch, the x86_64 assembly invokes
   `signal::delivery::kernel_post_syscall_signal_check(frame_ptr)`.
3. That function merges process-level and thread-level pending signals,
   removes blocked signals, picks the lowest pending signal, and calls
   `deliver_signal`.
4. `deliver_signal` either executes a default action (terminate, stop,
   continue, ignore) or builds a signal frame on the user stack and
   redirects the trap frame to the handler.

### Signal Frame (x86_64)

```
  User Stack (growing down)
  ┌─────────────────────────────────┐  ← new_rsp (16-byte aligned)
  │ trampoline[16]                  │  ← "mov $SYS_SIGRETURN, %eax; syscall; ud2"
  │ saved_rip, saved_rsp, saved_rflags │
  │ saved_rax … saved_r15           │  ← 16 general-purpose registers
  │ signum (u64)                    │
  │ saved_mask (u64)                │  ← signal mask before delivery
  │ _pad, _pad2 (u64 × 2)           │
  └─────────────────────────────────┘  192 bytes total
  ┌─────────────────────────────────┐  ← ret_addr_slot (new_rsp - 8)
  │ trampoline_addr (u64)           │  ← handler return address
  └─────────────────────────────────┘  ← actual user RSP at handler entry
```

The handler receives `signum` in `%rdi` (System V calling convention).  When
the handler executes `ret`, control goes to the trampoline, which executes
`syscall` with `eax = SYS_SIGRETURN`.

### sigreturn

`SYS_SIGRETURN` is handled specially in `kernel_dispatch_flat` before the
main dispatch table.  It calls `sys_sigreturn_inner(frame_ptr)`, which:

1. Reads the `SignalFrame` from the user stack (`frame_ptr + 8`).
2. Restores all 18 saved registers into the kernel trap frame.
3. Restores the saved signal mask via `set_signal_mask_current`.

---

## Kernel API

### Signal Generation

```rust
// Send to a process (signals all threads in the process's thread group).
crate::signal::send_signal_to_process(pid: u32, sig: u8) -> bool

// Send to a specific thread.
crate::signal::send_signal_to_thread(tid: u64, sig: u8)

// Notify parent on child exit/stop/continue.
crate::signal::notify_parent_sigchld(ppid: u32, child_pid: u32, status: i32)
```

### Alarm Tick Processing

`crate::signal::check_alarm_ticks(tick: u64)` must be called from the
scheduler timer interrupt to fire SIGALRM at the right tick.

---

## Syscall Interface

| Syscall        | Number  | Description                              |
|----------------|---------|------------------------------------------|
| `kill`         | 0x1013  | Send signal to process                   |
| `raise`        | 0x1014  | Send signal to calling thread            |
| `sigaction`    | 0x1015  | Set/query signal disposition             |
| `sigprocmask`  | 0x1016  | Set/query signal mask                    |
| `sigpending`   | 0x1017  | Query pending signals                    |
| `sigsuspend`   | 0x1018  | Atomically replace mask and wait         |
| `sigreturn`    | 0x1019  | Return from signal handler (arch)        |
| `alarm`        | 0x101A  | Schedule SIGALRM                         |
| `pause`        | 0x101B  | Wait for any signal                      |

### Userspace Wrappers (stem)

```rust
use stem::syscall::signal::*;

kill(pid, sig)                    // send signal to process
raise(sig)                        // send to self
sigaction(sig, Some(&act), None)  // install handler
sig_block(&set)                   // add to mask
sig_unblock(&set)                 // remove from mask
sig_setmask(&set)                 // replace mask
sigpending()                      // -> SigSet
sigsuspend(&mask)                 // -> Errno::EINTR always
alarm(seconds)                    // -> remaining seconds
pause()                           // -> Errno::EINTR always
```

---

## Shell Utility: `kill`

```
kill [-<signal>] <pid> [<pid> ...]
kill -l
```

Sends a signal to one or more processes.  The default signal is `SIGTERM`
(15).  Signal names and numbers are both accepted:

```sh
kill -9 1234          # SIGKILL pid 1234
kill -SIGTERM 1234    # SIGTERM pid 1234
kill -l               # list all signal names
```

---

## Behavioral Guarantees

| Behavior                              | Status     |
|---------------------------------------|------------|
| SIGKILL always terminates             | ✅          |
| SIGSTOP always stops (uncatchable)    | ✅          |
| SIGCONT resumes stopped process       | ✅          |
| Blocked signals remain pending        | ✅          |
| Ignored signals disappear             | ✅          |
| Handler installed with sigaction      | ✅          |
| Handlers run in userspace             | ✅ (x86_64) |
| sigreturn restores prior state        | ✅ (x86_64) |
| Signal mask updated during handler    | ✅          |
| SA_NODEFER respected                  | ✅          |
| SA_RESETHAND respected                | ✅          |
| pause() wakes on signal               | ✅          |
| sigsuspend() atomically swaps mask    | ✅          |
| SIGCHLD sent on child exit            | ✅          |
| SIGALRM via alarm()                   | ✅ (timer hook needed) |
| SIGPIPE on closed pipe write          | Planned     |
| SA_RESTART for blocked syscalls       | Planned     |
| Nested signal delivery                | Basic (depth not limited) |
| sigaltstack                           | Not implemented |
| Realtime signals (queued)             | Not implemented |
| Full siginfo_t                        | Not implemented |
| ptrace signal mediation               | Not implemented |
| Core dump files                       | Not implemented |

---

## Non-Goals (for this implementation)

The following are explicitly deferred:

- **Realtime signals**: `SIGRTMIN`–`NSIG-1` signal numbers are reserved in
  the bitmask but no queued delivery is implemented.
- **Full `siginfo_t`**: The `SigAction` struct has a `flags` field for
  `SA_SIGINFO` but three-argument handlers are not invoked.
- **`sigaltstack`**: Signal frames are always placed on the regular user stack.
- **Core dumps**: Signals with a "core dump" default action terminate the
  process normally without writing a file.
- **Session / TTY job control**: `SIGTTIN`/`SIGTTOU` default actions are
  implemented (stop), but full TTY session management is not.
- **ptrace signal injection/interception**.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `abi/src/signal.rs` | Signal constants, `SigSet`, `SigAction`, wait-status helpers |
| `abi/src/numbers.rs` | Syscall numbers for signal API |
| `kernel/src/signal/mod.rs` | `ProcessSignals`, `ThreadSignals`, generation helpers |
| `kernel/src/signal/delivery.rs` | x86_64 frame injection, sigreturn |
| `kernel/src/task/mod.rs` | `Thread.signals: ThreadSignals`, `Process.signals: ProcessSignals` |
| `kernel/src/sched/mod.rs` | SIGCHLD on child exit, signal hook registration |
| `kernel/src/sched/hooks.rs` | Type-erased hook accessors for signal mask |
| `kernel/src/syscall/handlers/signal.rs` | Syscall handler implementations |
| `kernel/src/syscall/flat.rs` | Post-syscall signal check invocation |
| `bran/src/arch/x86_64/syscall.rs` | Frame pointer pass-through to `kernel_dispatch_flat` |
| `stem/src/syscall/signal.rs` | Userspace syscall wrappers |
| `userspace/kill/src/main.rs` | `kill` shell utility |

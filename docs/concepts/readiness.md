# Unified Readiness Model

Thing-OS uses a single, coherent readiness model for all waitable kernel
objects.  The same set of flags, the same poll contract, and the same
blocking pattern apply whether you are waiting on a pipe, a channel, a
VFS-backed device thing, or a mix of all three.

---

## 1. Readiness Flags

All readiness is expressed as a bitmask using the constants in
`abi::syscall::poll_flags`:

| Flag       | Value  | Meaning |
|------------|--------|---------|
| `POLLIN`   | 0x0001 | Data is available to read (or EOF on a pipe/channel). |
| `POLLOUT`  | 0x0004 | Space is available to write without blocking. |
| `POLLERR`  | 0x0008 | An error condition is present; applicable to write ends when the peer has closed. |
| `POLLHUP`  | 0x0010 | The peer has closed its end (hangup). Always checked regardless of `events`. |
| `POLLNVAL` | 0x0020 | The thing is not open or is not valid. |

`POLLERR` and `POLLHUP` are always reported in `revents` if they occur,
even when not listed in `events`.

---

## 2. Per-Object-Class Semantics

### 2.1 Pipes (`PipeReadNode` / `PipeWriteNode`)

Pipes are anonymous byte streams created by `SYS_FS_PIPE`.  Each pipe has a
read end and a write end.

**Read end (`PipeReadNode`):**

| Condition | POLLIN | POLLHUP |
|-----------|--------|---------|
| Buffer has bytes | ✓ | — |
| Buffer empty, writer alive | — | — |
| Buffer has bytes, writer closed | ✓ | ✓ |
| Buffer empty, writer closed (EOF) | ✓ | ✓ |

A consumer detects EOF by receiving `POLLIN` with no readable bytes
(`read` returns 0) **or** by observing `POLLHUP` directly.

**Write end (`PipeWriteNode`):**

| Condition | POLLOUT | POLLERR | POLLHUP |
|-----------|---------|---------|---------|
| Buffer has free space, reader alive | ✓ | — | — |
| Buffer full, reader alive | — | — | — |
| Reader closed (broken pipe) | — | ✓ | ✓ |

A writer detects that the read end is gone by observing `POLLERR | POLLHUP`
on its write end.  A subsequent `write` call will return `EPIPE`.

### 2.2 Channels (`PortNode`)

Channels are bounded message queues created by `SYS_CHANNEL_CREATE`.  Each
channel exposes a read thing and a write thing, each of which can be
bridged to a VFS thing via `SYS_FS_FD_FROM_HANDLE`.

**Read thing:**

| Condition | POLLIN | POLLHUP |
|-----------|--------|---------|
| Queue has bytes | ✓ | — |
| Queue empty, writer alive | — | — |
| Queue empty, writer closed | ✓ | ✓ |
| Queue has bytes, writer closed | ✓ | ✓ |

**Write thing:**

| Condition | POLLOUT | POLLERR | POLLHUP |
|-----------|---------|---------|---------|
| Queue has free space, reader alive | ✓ | — | — |
| Queue full, reader alive | — | — | — |
| Reader closed | — | ✓ | ✓ |

The semantics mirror pipes exactly, which means the same event-loop code
can handle both without special-casing.

### 2.3 Unix Domain Sockets (`UnixSocketNode`)

Unix domain sockets (`SYS_SOCKET + AF_UNIX + SOCK_STREAM`) are bidirectional
VFS things.  Each connected end has its own read buffer (data written by the
peer) and tracks whether the peer has closed.

**Connected socket (read direction):**

| Condition | POLLIN | POLLHUP |
|-----------|--------|---------|
| Peer has written data | ✓ | — |
| No data, peer alive | — | — |
| Data available, peer closed | ✓ | ✓ |
| No data, peer closed (EOF) | ✓ | ✓ |

`POLLIN` is set whenever the local read buffer is non-empty **or** the peer has
closed (EOF).  A `vfs_read` that returns 0 signals EOF.

**Connected socket (write direction):**

| Condition | POLLOUT | POLLERR | POLLHUP |
|-----------|---------|---------|---------|
| Local write buffer has free space, peer alive | ✓ | — | — |
| Local write buffer full, peer alive | — | — | — |
| Peer closed (broken pipe) | — | ✓ | ✓ |

**Listening socket (server side):**

| Condition | POLLIN |
|-----------|--------|
| At least one connection in the accept queue | ✓ |
| Accept queue empty | — |

`POLLIN` on a listening socket means `accept()` will not block.

### 2.4 VFS-Backed Files

Regular things and device nodes opened via `SYS_FS_OPEN` implement
`VfsNode::poll`.  The default implementation returns `POLLIN | POLLOUT`
unconditionally, matching POSIX semantics for non-socket things:
**regular things are always ready**.

Device-specific implementations (e.g. a framebuffer driver, a terminal)
may override this to reflect actual buffer state.

---

## 3. The `SYS_FS_POLL` Syscall

```
SYS_FS_POLL(pollfds_ptr: usize, nfds: usize, timeout_ms: usize) -> SysResult<usize>
```

- `pollfds_ptr` — pointer to a `[PollFd; nfds]` array in user memory, where
                  each `PollFd.fd` is a **thing** number
- `nfds`        — number of entries (max 256)
- `timeout_ms`  — `0` = non-blocking; `usize::MAX` = block indefinitely;
                  any other value = timeout in milliseconds

Returns the number of entries with a non-zero `revents`, or an errno.

### Algorithm (no-lost-wakeup)

The kernel handler uses a three-phase algorithm to guarantee that a wakeup
produced between registration and the actual sleep is never lost:

1. **Probe** — call `VfsNode::poll()` on every entry.  If anything is
   ready or the timeout is 0, copy `revents` back and return immediately.
2. **Register** — call `VfsNode::add_waiter(tid)` on every node, then
   optionally arm a scheduler timeout via `register_timeout_wake_current`.
3. **Re-probe** — repeat the probe after registration.  If still nothing
   is ready, call `block_current_erased()` to park the task.
4. On wake, **unregister** from all nodes and re-probe to gather results.

This means the loop exits correctly whether the wakeup arrives before step 3
completes or after the task is parked.

---

## 4. Sample Event Loop

```rust
use stem::syscall::vfs::*;
use abi::syscall::{PollFd, poll_flags};

fn run_event_loop(pipe_read: u32, socket_thing: u32, channel_thing: u32) {
    let mut fds = [
        PollFd { fd: pipe_read as i32,       events: poll_flags::POLLIN, revents: 0 },
        PollFd { fd: socket_thing as i32,    events: poll_flags::POLLIN | poll_flags::POLLOUT, revents: 0 },
        PollFd { fd: channel_thing as i32,   events: poll_flags::POLLIN, revents: 0 },
    ];

    loop {
        let n = vfs_poll(&mut fds, u64::MAX).expect("poll failed");
        if n == 0 { continue; } // spurious wakeup

        for entry in &fds {
            if entry.revents == 0 { continue; }

            if entry.revents & poll_flags::POLLHUP != 0 {
                // Peer closed — handle EOF / broken pipe.
                break;
            }
            if entry.revents & poll_flags::POLLIN != 0 {
                // Read available data.
            }
            if entry.revents & poll_flags::POLLOUT != 0 {
                // Write space available.
            }
        }

        // Reset revents before next iteration.
        for entry in &mut fds { entry.revents = 0; }
    }
}
```

All three thing types — pipes, sockets, and channel things — are polled
through the same `SYS_FS_POLL` interface with identical flag semantics.

---

## 5. Peer-Death / Hangup Consistency

Hangup behaviour is uniform across all object classes:

- Closing the **write** end of a pipe or channel causes `POLLIN | POLLHUP`
  on the read end (the EOF signal).
- Closing the **read** end of a pipe or channel causes `POLLERR | POLLHUP`
  on the write end (broken pipe signal).
- Closing one end of a connected Unix socket causes `POLLIN | POLLHUP` on the
  peer's read direction (EOF) and `POLLERR | POLLHUP` on the peer's write
  direction (broken pipe).
- These bits are always reported in `revents` regardless of the `events`
  mask, so an event loop does not need to subscribe to them explicitly.

---

## 6. `SYS_WAIT_MANY` and `WaitKind::Fd`

`SYS_FS_POLL` is the preferred single-call interface for multiplexing many FDs.
For use cases that mix FDs with non-FD readiness sources (ports, task exit, IRQs),
the higher-level `SYS_WAIT_MANY` syscall (see `docs/wait_many.md`) accepts typed
`WaitSpec` entries.

The `WaitKind::Fd` variant routes through the same `VfsNode::poll` / waiter API:

```rust
use abi::wait::{WaitKind, WaitSpec, interest};

WaitSpec {
    kind: WaitKind::Fd as u32,
    flags: interest::READABLE,  // interest::WRITABLE for write readiness
    object: fd as u64,          // the file-descriptor number
    token: MY_TOKEN,
}
```

All VFS-backed resources — pipes, sockets, channel ends bridged with
`SYS_FS_FD_FROM_HANDLE`, and device nodes — use `WaitKind::Fd`.

### Task Exit

Task exit is observed through `WaitKind::TaskExit` rather than an FD today.
Future work may expose a pollable "process exit FD" that allows `SYS_FS_POLL`
loops to wait for child process termination alongside other I/O without a
separate `wait_many` call.

### Deprecated Kinds

`WaitKind::GraphOp` (= 6) and `WaitKind::RootWatch` (= 2) are **deprecated**
and return `ENOSYS`.  New code must not use them.  Existing binaries that pass
these kind values will receive a clean `ENOSYS` error.

---

## 7. Future Extensions

The same `VfsNode::poll` / `add_waiter` / `remove_waiter` contract is the
extension point for timer FDs, process-exit notification FDs, IRQ FDs, and
any other kernel waitable that needs to participate in `SYS_FS_POLL`.

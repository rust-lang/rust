# IPC Doctrine — Thing-OS Canonical Primitive Model

Thing-OS inter-process communication is built on six primitives.  Each has
one job.  Use the right primitive for the job; do not stretch one to cover
another.

> **Terminology note**: in Thing-OS a kernel-managed reference to an open
> object is called a **thing** — not a "file descriptor" (POSIX) and not a
> "handle" (Win32).  Everything is a thing.  Syscall return values that would
> be called `fd` in POSIX or `HANDLE` in Win32 are things here.  Code
> examples below use `thing` as the conventional variable name.

---

## 1. Primitive Overview

| Primitive | Syscall family | When to use |
|-----------|---------------|-------------|
| **Channel** | `SYS_CHANNEL_*` | Discrete messages: commands, ACKs, events, thing passing, RPC |
| **Pipe** | `SYS_PIPE` / `SYS_FS_*` | Sequential byte streams: stdio, process output pipelines |
| **Unix socket** | `SYS_SOCKET` / `SYS_BIND` / `SYS_CONNECT` / `SYS_SOCKETPAIR` | Bidirectional byte-stream endpoint IPC via filesystem path or anonymous pair |
| **Memfd** | `SYS_MEMFD_CREATE` / `SYS_VM_MAP` | Bulk data, zero-copy buffers, shared rings |
| **Futex** | `SYS_FUTEX_WAIT` / `SYS_FUTEX_WAKE` | Low-level in-process synchronisation, mutex/condvar building blocks |
| **Poll/wait** | `SYS_FS_POLL` | Readiness multiplexing across pipes, channels, sockets, and provider things |
| **VFS RPC** | `SYS_FS_MOUNT` + channel protocol | Structured request/reply filesystem provider interface |

---

## 2. Channels

### What channels are

A **channel** is a bounded, message-oriented, bidirectional-by-pair IPC
primitive.  `SYS_CHANNEL_CREATE` returns a write thing and a read thing that
share one ring buffer.

### Message model

- Messages are discrete: each `channel_send` / `channel_recv` pair transfers
  one logical unit.
- Maximum message size: 4 KiB (the ring capacity).  Larger payloads must use
  memfd (see §4).
- `channel_send_all` is atomic: it either writes the entire payload or fails
  with `EAGAIN`.  Prefer it over `channel_send` for protocol messages.

### Thing passing

A channel message may carry attached capability things via `channel_send_msg`.
The kernel re-numbers each thing in the receiver's thing table when the
receiver calls `channel_recv_msg`.  Supported capability types: VFS things,
memfds, channel endpoints, VFS provider things.

See `docs/concepts/channel_semantics.md` for the full specification.

### When to use channels

- Any control-plane exchange: service requests, device commands, event
  notifications, registration handshakes.
- Request/reply RPC that is not naturally file-shaped.
- Passing capabilities (things) between processes.

### When **not** to use channels

- Large binary blobs (images, audio, network payloads) — use memfd.
- Sequential byte streams without message boundaries — use a pipe.
- Shared-state synchronisation within a single address space — use a futex.

### Short example

```rust
use stem::syscall::channel::{channel_create, channel_send_all, channel_recv};

// Create a channel and send a one-shot command.
let (write_thing, read_thing) = channel_create(4096).expect("channel_create");
channel_send_all(write_thing, b"ping").expect("send");

let mut buf = [0u8; 8];
let n = channel_recv(read_thing, &mut buf).expect("recv");
assert_eq!(&buf[..n], b"ping");
```

---

## 3. Pipes

### What pipes are

A **pipe** is a one-way, anonymous byte stream.  `SYS_PIPE` returns a read
thing and a write thing backed by a kernel ring buffer.

### When to use pipes

- Parent–child stdio (thing 0, thing 1, thing 2).
- Any producer–consumer relationship where the data has no message boundaries
  and both ends run in a direct parent–child relationship.
- Shell pipelines.

### When **not** to use pipes

- RPC or structured message exchange — use a channel.
- Cross-service communication with capability passing — use a channel.
- Bulk one-shot data transfer — consider memfd.

See `docs/concepts/channels_vs_pipes.md` for a detailed comparison.

### Short example

```rust
use stem::syscall::vfs::{pipe, vfs_read, vfs_write, vfs_close};

// Create a pipe; write bytes on one end and read on the other.
let mut things = [0u32; 2];
pipe(&mut things).expect("pipe");
let (read_thing, write_thing) = (things[0], things[1]);

vfs_write(write_thing, b"hello").expect("write");
vfs_close(write_thing).expect("close write");   // signal EOF to reader

let mut buf = [0u8; 8];
let n = vfs_read(read_thing, &mut buf).expect("read");
assert_eq!(&buf[..n], b"hello");
vfs_close(read_thing).expect("close read");
```

---

## 4. Unix Domain Sockets

### What Unix domain sockets are

A **Unix domain socket** is a bidirectional, connection-oriented byte-stream
endpoint bound to a filesystem path under `/run` (or elsewhere).  The kernel
implements `AF_UNIX + SOCK_STREAM` sockets as first-class VFS things that
participate in the unified `SYS_FS_POLL` readiness model.

There are two usage modes:

- **Named** — a server binds to a path (`/run/foo.sock`), calls `listen()`, and
  accepts connections.  Clients call `connect("/run/foo.sock")`.  The bound path
  appears in the VFS as a socket file marker (`S_IFSOCK`).
- **Anonymous** — `SYS_SOCKETPAIR` creates a connected pair of socket things
  with no filesystem binding.  Useful for parent–child IPC before exec, or for
  intra-process thread communication.

Each connected socket thing supports `vfs_read`, `vfs_write`, `vfs_close`, and
`SYS_FS_POLL`, exactly like a pipe.

### When to use Unix sockets

- Server → client connections where multiple clients need to connect over time.
- Bidirectional peer-to-peer byte streams (both ends can read and write).
- Replacing a named pipe (FIFO) when bidirectionality is needed without a
  second pipe.
- Pre-exec channel setup between parent and child (use `socketpair()`).

### When **not** to use Unix sockets

- Unidirectional parent→child streams where `vfs_write` / `vfs_read` on a pipe
  suffice — a pipe is simpler.
- Structured message exchange or capability passing — use a channel.
- Bulk one-shot data transfer — use memfd.

### Short examples

**Named socket (server side):**

```rust
use stem::syscall::socket::{socket, bind, listen, accept};
use stem::syscall::vfs::{vfs_read, vfs_write};
use abi::syscall::{socket_domain::AF_UNIX, socket_type::SOCK_STREAM};

let srv = socket(AF_UNIX, SOCK_STREAM, 0).expect("socket");
bind(srv, "/run/echo.sock").expect("bind");
listen(srv, 8).expect("listen");

let conn = accept(srv).expect("accept");   // blocks until a client connects
let mut buf = [0u8; 256];
let n = vfs_read(conn, &mut buf).expect("read");
vfs_write(conn, &buf[..n]).expect("write"); // echo back
```

**Named socket (client side):**

```rust
use stem::syscall::socket::{socket, connect};
use stem::syscall::vfs::{vfs_write, vfs_read};
use abi::syscall::{socket_domain::AF_UNIX, socket_type::SOCK_STREAM};

let fd = socket(AF_UNIX, SOCK_STREAM, 0).expect("socket");
connect(fd, "/run/echo.sock").expect("connect");
vfs_write(fd, b"hello").expect("write");
let mut buf = [0u8; 8];
let n = vfs_read(fd, &mut buf).expect("read");
assert_eq!(&buf[..n], b"hello");
```

**Anonymous pair:**

```rust
use stem::syscall::socket::socketpair;
use stem::syscall::vfs::{vfs_write, vfs_read};
use abi::syscall::{socket_domain::AF_UNIX, socket_type::SOCK_STREAM};

let (a, b) = socketpair(AF_UNIX, SOCK_STREAM, 0).expect("socketpair");
vfs_write(a, b"ping").expect("write a");
let mut buf = [0u8; 8];
let n = vfs_read(b, &mut buf).expect("read b");
assert_eq!(&buf[..n], b"ping");
```

---

## 5. Memfd

### What memfd is

A **memfd** (`SYS_MEMFD_CREATE`) is an anonymous, resizable, in-kernel memory
object exposed as a thing.  The caller maps it into its address space
with `SYS_VM_MAP`.

### When to use memfd

- Pixel buffers, audio rings, network receive windows — any bulk-data path.
- Zero-copy exchange: sender writes into the mapped region, sends the thing
  over a channel; receiver maps it read-only.
- Persistent shared rings for audio or network I/O that avoids repeated thing
  transfers on the hot path.

### Doctrine

> **Control plane over channels; bulk data over memfd.**

Never embed multi-kilobyte payloads in channel messages.  Pass a
`abi::memfd::MemFdRef` (16 bytes) in the control message and the actual thing
via `channel_send_msg`.

See `docs/concepts/memfd.md` for the full lifecycle and wire format.

### Short example

```rust
use stem::syscall::memfd::memfd_create;
use stem::syscall::vm::{vm_map, MapFlags};
use stem::syscall::channel::channel_send_msg;

// Share a pixel buffer zero-copy via a memfd thing.
let frame_thing = memfd_create("frame", 1024 * 768 * 4).expect("memfd_create");
let ptr = vm_map(frame_thing, 0, 1024 * 768 * 4, MapFlags::READ_WRITE)
    .expect("vm_map");
// … fill pixel data at ptr …
channel_send_msg(chan_write_thing, b"frame-ready", &[frame_thing])
    .expect("send_msg");
// Receiver calls channel_recv_msg → gets a new thing for the same memfd region.
```

---

## 6. Futex

### What futex is

A **futex** is a 32-bit user-space integer at a known virtual address.  The
kernel provides `SYS_FUTEX_WAIT` and `SYS_FUTEX_WAKE` to park or wake threads
based on the value of that integer.  Futexes are not things — they are raw
memory addresses within the caller's address space.

### When to use futex

- Building mutex / condvar / semaphore primitives in user space.
- Spinning-then-blocking patterns in lock-free data structures.
- Any in-process synchronisation where `stem::sync` primitives already wrap
  futexes correctly.

### When **not** to use futex

- Cross-process synchronisation that does not share a memory mapping — use
  channels.
- Any IPC that carries data, not just a signal — use a channel or pipe.

### Short example

```rust
use core::sync::atomic::{AtomicU32, Ordering};
use stem::syscall::futex::{futex_wait, futex_wake};

static LOCK: AtomicU32 = AtomicU32::new(0); // 0 = unlocked, 1 = locked

// Acquire: spin once, then block if still held.
while LOCK.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
    futex_wait(LOCK.as_ptr(), 1, None).ok(); // park until value changes
}

// Release: clear the lock and wake one waiter.
LOCK.store(0, Ordering::Release);
futex_wake(LOCK.as_ptr(), 1).ok();
```

---

## 7. Poll/wait

### What poll/wait is

`SYS_FS_POLL` is the unified readiness multiplexer.  Given a list of things
(pipes, channels bridged via `SYS_FD_FROM_HANDLE`, or provider things) and
an optional timeout, it returns the set of things that are ready for the
requested I/O operations.

### When to use poll/wait

- Any event loop that needs to wait on more than one source at once.
- Implementing `select`/`epoll`-style reactor patterns.
- Waiting for the first of several channel replies to arrive.

### When **not** to use poll/wait

- Waiting on a single thing that you can block on directly — just call
  `channel_recv` or `vfs_read`; both park the thread until data arrives.
- In-process signalling between threads — use a futex.

### Short example

```rust
use stem::syscall::vfs::fs_poll;
use abi::poll::{PollItem, POLLIN};

// Wait on a pipe read thing and a channel read thing simultaneously.
let items = [
    PollItem { thing: pipe_read_thing, events: POLLIN },
    PollItem { thing: chan_read_thing, events: POLLIN },
];
let ready_count = fs_poll(&mut items.clone(), u64::MAX /* block forever */)
    .expect("poll");
for item in &items[..ready_count] {
    if item.revents & POLLIN != 0 {
        // … read from item.thing …
    }
}
```

See `docs/concepts/readiness.md` for the full readiness flag semantics.

---

## 8. VFS RPC

### What VFS RPC is

VFS RPC allows a userland process to export a subtree of the filesystem
namespace.  The kernel serialises every filesystem operation (open, read,
write, stat, readdir …) into a typed message and delivers it to the provider's
channel thing.  The provider answers synchronously.

### When to use VFS RPC

- Implementing filesystem drivers (iso9660d, network FS, synthetic /proc entries).
- Exposing a device as a set of things under `/dev`.
- Any service that looks naturally file-shaped to its clients.

### When **not** to use VFS RPC

- Low-latency event sources — use a channel directly.
- Bulk streaming — combine VFS RPC with memfd for the data path.

See `docs/concepts/vfs_rpc_provider.md` for the provider lifecycle and wire
protocol.

### Short example

```rust
use ipc_helpers::provider::{ProviderLoop, ProviderResponse};
use abi::vfs_rpc::VfsRpcOp;

// Mount a virtual directory at /run/myservice and answer open/read ops.
let (provider_write_thing, provider_read_thing) = channel_create(4096)
    .expect("channel_create");
fs_mount("/run/myservice", provider_write_thing).expect("mount");

let mut loop_ = ProviderLoop::new(provider_read_thing);
loop_.run(|op| match op {
    VfsRpcOp::Open { name, .. } => ProviderResponse::ok_thing(make_file_thing(name)),
    VfsRpcOp::Read { thing, buf, .. } => ProviderResponse::data(read_file(thing, buf)),
    _ => ProviderResponse::err(Errno::ENOSYS),
});
```

---

## 9. Decision Matrix

| I need to … | Use |
|-------------|-----|
| Send a command / event to a service | Channel |
| Do synchronous request/reply RPC | Channel + `abi::rpc::RpcHeader` |
| Pass a thing (capability) to another process | Channel + `channel_send_msg` |
| Transfer a large buffer zero-copy | Memfd thing + channel (to pass the thing) |
| Stream bytes parent→child (unidirectional) | Pipe |
| Bidirectional byte-stream between two processes | Unix socket (`socketpair` or named) |
| Service accepting connections from many clients | Unix socket (named, `bind`+`listen`+`accept`) |
| Expose a subtree as a filesystem | VFS RPC |
| Wait on multiple things at once | `SYS_FS_POLL` (works for all VFS things) |
| Implement a mutex or condvar | Futex (or `stem::sync` wrappers) |

---

## 10. See Also

- `docs/concepts/channel_semantics.md` — capacity, atomicity, blocking, peer death
- `docs/concepts/channels_vs_pipes.md` — when to use which
- `docs/concepts/vfs_rpc_provider.md` — provider lifecycle
- `docs/concepts/memfd.md` — bulk-data path
- `docs/concepts/readiness.md` — poll/wait model (covers pipes, channels, and sockets)
- `docs/concepts/supervisor_protocol.md` — service registration over channels
- `docs/concepts/ipc_cookbook.md` — practical recipes
- `abi/src/rpc.rs` — structured request/reply header types
- `libs/ipc_helpers/` — userspace helper library

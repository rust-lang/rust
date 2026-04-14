# IPC Cookbook — Practical Recipes for Driver and Service Authors

This cookbook shows how to use the Thing-OS IPC primitives for the most common
patterns.  Each recipe is self-contained.  Prerequisites: read
`docs/concepts/ipc.md` first.

---

## Recipe 1 — Parent/child stdio via pipe

**Problem**: create a pipe and stream data through it.

See `userspace/ipc_pipe_demo/` for the complete compilable example.

### Single-process pipe round-trip

```rust
use stem::syscall::vfs::{pipe, vfs_close, vfs_read, vfs_write};

fn pipe_round_trip() {
    let mut pipefds = [0u32; 2];
    pipe(&mut pipefds).expect("pipe");
    let (pr, pw) = (pipefds[0], pipefds[1]);

    // Write data to the write-end.
    vfs_write(pw, b"Hello!\n").expect("write");

    // Close the write-end so the reader sees EOF.
    vfs_close(pw).expect("close write");

    // Drain the read-end.
    let mut buf = [0u8; 256];
    loop {
        match vfs_read(pr, &mut buf) {
            Ok(0) | Err(_) => break,   // EOF or error
            Ok(n) => {
                let s = core::str::from_utf8(&buf[..n]).unwrap_or("?");
                stem::println!("{}", s);
            }
        }
    }
    vfs_close(pr).unwrap();
}
```

### Capturing child stdout via `spawn_process_ex`

To capture a child process's stdout, use `spawn_process_ex` with
`stdout_mode = PIPE`.  The kernel creates the pipe and gives the parent the
read end via `resp.stdout_pipe`.

```rust
use alloc::collections::BTreeMap;
use abi::types::stdio_mode;
use stem::syscall::{spawn_process_ex, vfs_read};

fn capture_child_stdout() {
    let resp = spawn_process_ex(
        "my_child",
        &[],
        &BTreeMap::new(),
        stdio_mode::INHERIT,  // stdin: inherit
        stdio_mode::PIPE,     // stdout: pipe — parent gets the read end
        stdio_mode::INHERIT,  // stderr: inherit
        0,                    // boot_arg
        &[],                  // extra handles
    )
    .expect("spawn");

    // resp.stdout_pipe is the parent's read end of the child's stdout pipe.
    let pr = resp.stdout_pipe as u32;

    let mut buf = [0u8; 256];
    loop {
        match vfs_read(pr, &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let s = core::str::from_utf8(&buf[..n]).unwrap_or("?");
                stem::println!("{}", s);
            }
        }
    }
}
```

---

## Recipe 2 — Request/reply service over a channel

**Problem**: expose a service that answers typed requests and sends typed
replies.

Use the `ipc_helpers::rpc` helpers instead of assembling `RpcHeader` bytes
by hand.  The helpers manage correlation-ID assignment, encoding, and
decoding automatically.

### Server side

```rust
use ipc_helpers::rpc::RpcServer;

fn run_service(read_h: u32, write_h: u32) {
    // read_h  — channel read end (server receives requests)
    // write_h — channel write end (server sends replies)
    let mut server = RpcServer::new(read_h);
    loop {
        let req = match server.next() {
            Ok(r) => r,
            Err(_) => continue, // channel error; yield and retry
        };
        // req.request_id — correlation ID to echo back
        // req.payload    — application payload (after the RpcHeader)

        let reply_payload: &[u8] = b"ok";
        server.reply(req.request_id, write_h, reply_payload).ok();
    }
}
```

### Client side

```rust
use ipc_helpers::rpc::RpcClient;

fn call_service(req_write_h: u32, rep_read_h: u32) {
    let client = RpcClient::new(req_write_h, rep_read_h);

    // call() sends the payload as a request and blocks until the matching
    // reply arrives.  Correlation is handled automatically.
    let reply = client.call(b"hello").expect("RPC failed");
    stem::println!("reply: {:?}", reply);
}
```

### Channel setup (in the supervisor / launcher)

```rust
use stem::syscall::channel::channel_create;

// Create a paired channel:
//   req_chan: clients write requests → server reads
//   rep_chan: server writes replies  → clients read
let req_chan = channel_create(4096).unwrap(); // (req_write, req_read)
let rep_chan = channel_create(4096).unwrap(); // (rep_write, rep_read)

// Pack both server handles into a single usize for spawn_process:
//   server receives arg0 = (rep_write << 16) | req_read
let server_arg = ((rep_chan.0 as usize) << 16) | (req_chan.1 as usize);
spawn_process("/bin/my_service", server_arg).unwrap();

// Clients use req_chan.0 (write) to send and rep_chan.1 (read) to receive.
let client = RpcClient::new(req_chan.0, rep_chan.1);
```

> **Raw API** (for reference / low-level use only): See the golden-bytes
> example in `abi/src/rpc.rs` tests or the pre-helper pattern shown below.

<details>
<summary>Raw RpcHeader approach (not recommended for new code)</summary>

```rust
use stem::syscall::channel::{channel_recv, channel_send_all};
use abi::rpc::{RpcHeader, RPC_FLAG_REPLY};

fn run_service_raw(read_h: u32, write_h: u32) {
    let mut buf = [0u8; 512];
    loop {
        let n = channel_recv(read_h, &mut buf).expect("recv");
        if n < RpcHeader::WIRE_SIZE { continue; }
        let hdr = RpcHeader::decode_le(&buf[..RpcHeader::WIRE_SIZE]).unwrap();
        let _payload = &buf[RpcHeader::WIRE_SIZE..n];

        let reply_hdr = RpcHeader {
            request_id: hdr.request_id,
            flags: RPC_FLAG_REPLY,
            _pad: [0; 5],
        };
        let mut out = [0u8; RpcHeader::WIRE_SIZE + 2];
        reply_hdr.encode_le(&mut out[..RpcHeader::WIRE_SIZE]).unwrap();
        out[RpcHeader::WIRE_SIZE..].copy_from_slice(b"ok");
        channel_send_all(write_h, &out).ok();
    }
}
```

</details>

---

## Recipe 3 — Handle passing (first-class message API)

**Problem**: transfer one or more things (capabilities) from one
process to another as part of a message.

### New API (preferred): `channel_send_msg` / `channel_recv_msg`

Handles are **first-class properties of a message** — they travel atomically
alongside the data bytes in the same message unit.

```rust
use stem::syscall::channel::{channel_send_msg, channel_recv_msg};

// Sender: send data bytes + two fds in one atomic operation.
fn send_fds(channel: u32, fd1: u32, fd2: u32) {
    let handles = [fd1, fd2];
    channel_send_msg(channel, b"two-fds", &handles).expect("send_msg");
}

// Receiver: receive data + fds together.
fn recv_fds(channel: u32) -> (u32, u32) {
    let mut data = [0u8; 16];
    let mut new_fds = [0u32; 2];
    let (data_len, handles_count) =
        channel_recv_msg(channel, &mut data, &mut new_fds).expect("recv_msg");
    assert_eq!(&data[..data_len], b"two-fds");
    assert_eq!(handles_count, 2);
    (new_fds[0], new_fds[1])
}
```

The kernel re-numbers each thing in the receiver's thing table.  Duplicate
semantics: the sender retains its own thing.

### Legacy API (deprecated): `channel_send_handle` / `channel_recv_handle`

> **⚠ Deprecated**: Use `channel_send_msg` / `channel_recv_msg` instead.  The
> old single-handle API is retained only for backward compatibility.  New code
> must use the atomic FD-passing APIs shown in the section above.

The old single-thing API is still supported as a compatibility wrapper:

```rust
use stem::syscall::channel::{channel_recv_msg, channel_send_msg, channel_send_all, channel_recv};

// Sender (FD-first):
fn send_fd(channel: u32, fd: u32) {
    channel_send_msg(channel, b"fd-ready", &[fd]).expect("send_msg");
}

// Receiver (FD-first):
fn recv_fd(channel: u32) -> u32 {
    let mut tag = [0u8; 8];
    let mut fds = [0u32; 1];
    let (_, n_fds) = channel_recv_msg(channel, &mut tag, &mut fds).expect("recv_msg");
    assert_eq!(&tag, b"fd-ready");
    assert_eq!(n_fds, 1);
    fds[0]
}
```

The kernel re-numbers the fd in the receiver's fd table.  The receiver can
use the received fd with any `SYS_FS_*` syscall immediately.

> **Note**: The `channel_send_msg` / `channel_recv_msg` API bundles data and
> FDs atomically in a single message, eliminating ordering races between the
> data queue and capability queue that existed in the old split API.

---

## Recipe 4 — VFS provider implementation

**Problem**: expose a virtual directory tree under a mount point.

See `userspace/ipc_provider_demo/` for the complete compilable example,
`libs/ipc_helpers/src/provider.rs` for the `ProviderLoop` helper, and
`userspace/iso9660d/` for a full-featured reference implementation.

Minimal skeleton:

```rust
use ipc_helpers::provider::{ProviderLoop, ProviderResponse};
use abi::vfs_rpc::VfsRpcOp;
use abi::errors::Errno;
use stem::syscall::{channel_create, vfs_mount};

fn run_provider_service() {
    // 1. Create the provider channel pair (write_handle, read_handle).
    let (write_h, read_h) = channel_create(65536).expect("channel_create");

    // 2. Mount: hand the write-end to the kernel.
    vfs_mount(write_h, "/run/myprovider").expect("vfs_mount");

    // 3. Serve VFS RPC requests.
    run_provider(read_h);
}

fn run_provider(vfs_read: u32) {
    let mut lp = ProviderLoop::new(vfs_read);
    loop {
        let req = match lp.next_request() {
            Ok(r) => r,
            Err(_) => break, // channel closed, shutdown
        };
        let resp = match req.op {
            VfsRpcOp::Lookup => {
                let path = core::str::from_utf8(&req.payload[4..]).unwrap_or("");
                if path == "hello.txt" {
                    ProviderResponse::ok_u64(1) // handle = 1
                } else {
                    ProviderResponse::err(Errno::ENOENT)
                }
            }
            VfsRpcOp::Read => ProviderResponse::ok_read(b"Hello, world!\n"),
            VfsRpcOp::Stat => ProviderResponse::ok_stat(0o100644, 14, 1),
            VfsRpcOp::Close => ProviderResponse::ok_empty(),
            _ => ProviderResponse::err(Errno::ENOSYS),
        };
        lp.send_response(req.resp_port, resp).ok();
    }
}
```

---

## Recipe 5 — Memfd-backed shared buffer exchange

**Problem**: transfer a large pixel buffer from a display driver to a compositor
without copying.

See `userspace/ipc_memfd_demo/` for the complete compilable example.

```rust
use stem::syscall::{memfd_create, vm_map, vm_unmap, vfs_close};
use stem::syscall::channel::{channel_send_msg, channel_recv_msg, channel_recv};
use abi::memfd::{MemFdRef, MEMFD_REF_WIRE_SIZE};
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};

const W: usize = 1920;
const H: usize = 1080;
const BPP: usize = 4;
const SIZE: usize = W * H * BPP;

// Sender (display driver):
fn send_frame(channel: u32) {
    let fd = memfd_create("frame", SIZE).unwrap();
    let req = VmMapReq {
        addr_hint: 0,
        len: SIZE,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::empty(),
        backing: VmBacking::File { fd, offset: 0 },
    };
    let mapped = vm_map(&req).unwrap();
    let pixels = unsafe {
        core::slice::from_raw_parts_mut(mapped.addr as *mut u32, W * H)
    };
    // … render into pixels …
    let _ = pixels; // suppress unused warning

    // Transfer the fd + descriptor in one atomic message.
    let desc = MemFdRef::new(fd, SIZE as u64);
    let handles = [fd];
    let mut ctrl = [0u8; 1 + MEMFD_REF_WIRE_SIZE];
    ctrl[0] = 0x01; // MSG_PRESENT
    desc.encode_le(&mut ctrl[1..]).unwrap();
    channel_send_msg(channel, &ctrl, &handles).unwrap();

    vm_unmap(mapped.addr, SIZE).unwrap();
    // vfs_close(fd) would free physical pages once receiver also closes
}

// Receiver (compositor):
fn recv_frame(channel: u32) {
    let mut ctrl = [0u8; 1 + MEMFD_REF_WIRE_SIZE];
    let mut handles = [0u32; 1];
    let (_, n_handles) = channel_recv_msg(channel, &mut ctrl, &mut handles).unwrap();
    assert_eq!(n_handles, 1);
    let new_fd = handles[0];

    let desc = MemFdRef::decode_le(&ctrl[1..]).unwrap();
    let req = VmMapReq {
        addr_hint: 0,
        len: desc.length as usize,
        prot: VmProt::READ | VmProt::USER,
        flags: VmMapFlags::empty(),
        backing: VmBacking::File { fd: new_fd, offset: 0 },
    };
    let mapped = vm_map(&req).unwrap();
    let pixels = unsafe {
        core::slice::from_raw_parts(mapped.addr as *const u32, W * H)
    };
    // … consume pixels …
    let _ = pixels; // suppress unused warning
    vm_unmap(mapped.addr, desc.length as usize).unwrap();
    vfs_close(new_fd).unwrap();
}
```

---

## Recipe 6 — Poll-based event loop

**Problem**: wait on a channel, a pipe read end, and a device file all at once.

See `userspace/poll_mux/` for the complete compilable example.

```rust
use stem::syscall::vfs::{vfs_poll, vfs_fd_from_handle};
use abi::syscall::{PollFd, poll_flags};

fn event_loop(pipe_read: u32, channel_write_h: u32, channel_read_h: u32, dev_fd: u32) {
    // Bridge the channel read thing into a VFS thing for poll.
    let channel_fd = vfs_fd_from_handle(channel_read_h).expect("bridge");

    let mut fds = [
        PollFd { fd: pipe_read as i32,  events: poll_flags::POLLIN, revents: 0 },
        PollFd { fd: channel_fd as i32, events: poll_flags::POLLIN, revents: 0 },
        PollFd { fd: dev_fd as i32,     events: poll_flags::POLLIN, revents: 0 },
    ];

    loop {
        let n = vfs_poll(&mut fds, u64::MAX).expect("poll");
        if n == 0 { continue; }

        for entry in &fds {
            if entry.revents == 0 { continue; }
            if entry.revents & poll_flags::POLLHUP != 0 {
                // Peer closed — clean up and exit.
                return;
            }
            if entry.revents & poll_flags::POLLIN != 0 {
                // Read available data.
                let mut buf = [0u8; 256];
                stem::syscall::vfs::vfs_read(entry.fd as u32, &mut buf).ok();
            }
        }

        for entry in &mut fds { entry.revents = 0; }
    }
}
```

---

## Recipe 7 — Supervisor startup/registration handshake

**Problem**: register a new driver with the supervisor so it gets a path under
`/dev`.

See `docs/concepts/supervisor_protocol.md` for the full specification and
`drivers/display_bootfb/src/main.rs` for a complete reference implementation.

Minimal sketch:

```rust
use stem::syscall::channel::{channel_create, channel_send_msg, channel_recv_msg};
use abi::supervisor_protocol::{self, classes};
use abi::vfs_rpc::VFS_RPC_MAX_REQ;

fn register_driver(drv_req_read: u32, drv_resp_write: u32, bind_instance_id: u64) {
    // 1. Create provider channel.
    let (vfs_write, vfs_read) = channel_create(VFS_RPC_MAX_REQ * 8).unwrap();

    // 2. Send provider FD + BIND_READY atomically (FD-first).
    let payload = supervisor_protocol::BindReadyPayload {
        bind_instance_id,
        class_mask: classes::DISPLAY_CARD | classes::FRAMEBUFFER,
        _reserved: 0,
    };
    let mut payload_bytes = [0u8; supervisor_protocol::BIND_READY_PAYLOAD_SIZE];
    supervisor_protocol::encode_bind_ready_le(&payload, &mut payload_bytes);

    // Bundle BIND_READY data and the provider FD atomically.
    channel_send_msg(drv_resp_write, &payload_bytes, &[vfs_write]).unwrap();

    // 3. Wait for BIND_ASSIGNED (received via channel_recv_msg).
    let mut buf = [0u8; 256];
    let mut fds = [0u32; 0];
    loop {
        if let Ok((n, _)) = channel_recv_msg(drv_req_read, &mut buf, &mut fds) {
            // parse msg_type, check for MSG_BIND_ASSIGNED …
            break;
        }
        stem::syscall::yield_now();
    }

    // 4. Serve VFS requests on vfs_read.
    run_provider(vfs_read);
}
```

---

## See Also

- `docs/concepts/ipc.md` — primitive overview
- `docs/concepts/channel_semantics.md` — channel specification
- `docs/concepts/channels_vs_pipes.md` — channel vs pipe
- `docs/concepts/vfs_rpc_provider.md` — VFS provider lifecycle
- `docs/concepts/memfd.md` — memfd bulk-data path
- `docs/concepts/readiness.md` — poll/wait model
- `docs/concepts/supervisor_protocol.md` — registration protocol
- `libs/ipc_helpers/` — userspace helper library
- `abi/src/rpc.rs` — request/reply header types

## Compilable Example Programs

Each recipe has a dedicated, runnable example that ships with the OS image:

| Recipe | Example binary | Source path |
|--------|---------------|-------------|
| 1 — pipe stdio | `ipc_pipe_demo` | `userspace/ipc_pipe_demo/` |
| 2 — RPC service | `ipc_service_demo` | `userspace/ipc_service_demo/` |
| 4 — VFS provider | `ipc_provider_demo` | `userspace/ipc_provider_demo/` |
| 5 — memfd buffer | `ipc_memfd_demo` | `userspace/ipc_memfd_demo/` |
| 6 — poll event loop | `poll_mux` | `userspace/poll_mux/` |
| 7 — supervisor handshake | `drivers/display_bootfb` | `drivers/display_bootfb/` |

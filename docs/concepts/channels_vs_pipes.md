# Channels vs Pipes — When to Use Which

Both channels and pipes move bytes between processes, but they serve
fundamentally different purposes.  Mixing them up leads to awkward protocol
code or unnecessary complexity.

---

## 1. Side-by-Side Comparison

| Property | Channel | Pipe |
|----------|---------|------|
| **Byte model** | Discrete messages | Continuous byte stream |
| **Message boundaries** | Preserved | Not preserved |
| **Direction** | Unidirectional pair (one write thing + one read thing) | One-way (read thing + write thing) |
| **Capacity** | Configurable ring, 64 B – 64 KiB | Fixed kernel ring (4 KiB default) |
| **Thing passing** | Yes — `channel_send_msg` / `channel_recv_msg` | No |
| **Plane** | Control plane — commands, ACKs, events, RPC | Data plane — raw byte streams |
| **Typical use** | Service requests, events, RPC, capability transfer | stdio, process output pipelines |
| **Poll integration** | Yes — bridge with `SYS_FD_FROM_HANDLE` | Yes — read/write ends are VFS things |
| **Syscall family** | `SYS_CHANNEL_*` | `SYS_PIPE` + `SYS_FS_*` |

---

## 2. Decision Rules

### Use a **channel** when

- You are sending discrete protocol messages (e.g. commands, ACKs, event
  notifications).
- You need to pass a capability (memfd, provider thing) to another process.
- You are implementing request/reply RPC with a service.
- You need bidirectional communication without creating two separate pipes.
- You want to wait on the channel together with other VFS things in one `poll`
  call.

### Use a **pipe** when

- You are wiring up stdio (thing 0, thing 1, thing 2) for a child process.
- You have a shell pipeline: the output of one process feeds the input of
  another.
- The data has no message boundaries (pure byte stream: text, binary output).
- You just need a one-way data stream with no metadata or capability transfer.

---

## 3. Poll Semantics Differences

The poll event flags for pipes and bridged channels are intentionally identical
so that the same event-loop code works with both.  However, the _conceptual_
difference matters:

| Event | Pipe read thing | Channel read thing (bridged) |
|-------|-----------------|------------------------------|
| Data available | `POLLIN` | `POLLIN` |
| Writer closed (EOF) | `POLLIN \| POLLHUP` | `POLLIN \| POLLHUP` |

| Event | Pipe write thing | Channel write thing (bridged) |
|-------|------------------|-------------------------------|
| Space available | `POLLOUT` | `POLLOUT` |
| Reader closed | `POLLERR \| POLLHUP` | `POLLERR \| POLLHUP` |

**Pipe**: `POLLIN` means "bytes are available in the stream".  The consumer
does not know where one "message" ends and the next begins — the stream is
continuous.

**Channel** (bridged): `POLLIN` means "at least one byte of a message is
available in the ring".  The consumer should call `channel_recv` (or
`channel_recv_msg`) to retrieve all bytes of the current message before
polling again.  If the consumer treats channel data as a byte stream (calling
`vfs_read` on the bridge fd in a loop), it will silently split messages across
read boundaries.

---

## 4. Common Misconceptions

### "I'll use a pipe because it's simpler"

Pipes do not support thing passing.  If your protocol ever needs to transfer
a capability thing, you must use a channel.  Starting with a channel avoids a
later refactor.

### "I'll use a channel for stdio"

Channels are not byte streams.  Each `channel_send` is a discrete message.  A
`cat` reading from a channel thing would receive one message per `channel_send`
call, not a continuous stream.  Use a pipe for stdio.

### "Pipes are lower overhead"

For small kernel-resident workloads both are comparable.  Channels add the
overhead of the thing table lookup; pipes add a VFS open-flags check.  Neither
difference is meaningful at application level.

### "I can use a channel as a transport for my own message framing"

You can — `channel_send` supports partial writes like a byte stream — but you
**should not**.  Adding your own framing layer on top of a channel is the same
work as writing a pipe protocol, with extra overhead.  If you find yourself
writing a `FrameReader` or re-assembling messages from successive `channel_recv`
calls, consider using `channel_send_msg` / `channel_recv_msg` (which preserves
your message as a single atomic unit) or switching to a pipe.

---

## 5. Migration Guide

### Pipe used for message protocol → channel

If you have existing code that uses a pipe for a structured message protocol:

1. Replace `SYS_PIPE` with `SYS_CHANNEL_CREATE`.
2. Replace `SYS_FS_WRITE(write_thing, …)` with `SYS_CHANNEL_SEND_ALL(write_thing, …)`.
3. Replace `SYS_FS_READ(read_thing, …)` with `SYS_CHANNEL_RECV(read_thing, …)`.
4. If you bridge the things to VFS things for poll, call `SYS_FD_FROM_HANDLE` on
   each thing after creation.

### Channel used for byte stream → pipe

If you have existing code that uses a channel for raw byte streaming (e.g.
streaming PCM audio, text output) and you want to migrate to a pipe:

1. Create a pipe with `SYS_PIPE` and retain `(pipe_read_fd, pipe_write_fd)`.
2. Transfer the write end to the consumer with
   `SYS_CHANNEL_SEND_HANDLE(ctrl_channel, pipe_write_fd)`.
3. The consumer retrieves the write fd with
   `SYS_CHANNEL_RECV_HANDLE(ctrl_channel)` and uses `SYS_FS_WRITE` for data.
4. The producer uses `SYS_FS_READ(pipe_read_fd, …)` to read the stream.

> **Note**: pipe FDs cannot be shared across processes by embedding them in
> a file or a plain integer field — they require capability transfer via
> `channel_send_handle`.  Design your service discovery to use a control channel
> for the initial connection and capability handoff.

---

## 6. Known Legacy Deviations

The following areas of the codebase currently use channels for raw byte
streaming.  This is a known anti-pattern; the code is annotated with FIXME
comments and tracked in
[issue #591](https://github.com/dancxjo/thing-os/issues/591).

| Component | File | What it does wrong | Correct approach |
|-----------|------|--------------------|-----------------|
| `beeper` | `drivers/beeper/src/main.rs` | Sends raw PCM audio chunks over a channel | Pipe (or memfd ring) for the PCM stream; channel for discovery/control |
| `hdaudio` | `drivers/hdaudio/src/main.rs` | Receives raw PCM audio chunks via `channel_recv` | Pipe (or memfd ring) for the PCM stream; channel for discovery/control |
| `virtio_sound` | `drivers/virtio_sound/src/main.rs` | Same as hdaudio | Same |

The root cause is that `AudioInfoPayload` stores a bare channel handle number
in a VFS file, which makes it readable by any process without a capability
transfer.  A proper pipe-based design requires a `channel_send_handle` step to
deliver the pipe write-end to the connecting client.

---

## 7. See Also

- `docs/concepts/ipc.md` — full primitive overview and decision matrix
- `docs/concepts/channel_semantics.md` — channel specification
- `docs/concepts/readiness.md` — unified poll/wait model
- `abi/src/numbers.rs` — `SYS_CHANNEL_*` and `SYS_PIPE` syscall numbers
- `kernel/src/ipc/pipe.rs` — pipe implementation
- `kernel/src/ipc/port.rs` — channel (port) implementation

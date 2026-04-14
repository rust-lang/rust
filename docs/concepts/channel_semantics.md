# Channel Semantics Specification

This document is the authoritative specification for the channel primitive in
Thing-OS.  Every kernel implementation detail referenced here is in
`kernel/src/ipc/port.rs` and `kernel/src/syscall/handlers/port.rs`.

---

## 1. What Is a Channel

A **channel** is a bounded, FIFO, byte-oriented ring buffer shared between
exactly one writer thing and one reader thing.

`SYS_CHANNEL_CREATE(capacity) -> (write_thing, read_thing)`

- `capacity` is clamped to `[64, 65536]` bytes and rounded up to the next
  power of two.
- The kernel returns a packed `usize`: `(write_thing << 16) | read_thing`.
- Thing `0` is reserved and always invalid.

---

## 2. Capacity and Message Size

| Limit | Value | Notes |
|-------|-------|-------|
| Minimum ring capacity | 64 bytes | |
| Maximum ring capacity | 65536 bytes (64 KiB) | Requested via `SYS_CHANNEL_CREATE` |
| Maximum single message | 4096 bytes (4 KiB) | Enforced by `SYS_CHANNEL_SEND` / `SYS_CHANNEL_SEND_ALL` |
| Maximum attached things per `SYS_CHANNEL_SEND_MSG` | 64 | Returns `EINVAL` if exceeded |
| Maximum attached things per `SYS_CHANNEL_SEND_HANDLE` | 1 | Queued independently from byte data |

> **Practical guideline**: protocol messages should fit in a few hundred bytes.
> For anything larger, embed a `abi::memfd::MemFdRef` and transfer the data
> via memfd.

---

## 3. Atomicity

### `SYS_CHANNEL_SEND` (partial write allowed)

- Writes as many bytes as the available ring space allows.
- Returns the number of bytes actually written (may be less than `len`).
- Returns `EPIPE` if the read end is closed.

### `SYS_CHANNEL_SEND_ALL` (all-or-nothing)

- If `len` bytes fit in the ring, all are written atomically.
- If `len` bytes do **not** fit, **no bytes** are written.
- Returns `EAGAIN` when the ring is full (not a partial failure).
- Use `SYS_CHANNEL_SEND_ALL` for protocol messages that must not be split.

### `SYS_CHANNEL_RECV` (blocking)

- Reads up to `len` bytes from the ring.
- Blocks until at least 1 byte is available.
- Returns `EPIPE` if the write end is closed and the ring is empty.

### `SYS_CHANNEL_TRY_RECV` (non-blocking)

- Like `SYS_CHANNEL_RECV` but returns `EAGAIN` immediately if the ring is
  empty instead of blocking.

---

## 4. FIFO Guarantee

Bytes arrive at the receiver in the same order they were written by the sender.
There is no reordering.

---

## 5. Blocking Behaviour

### Sender blocks / returns early

`SYS_CHANNEL_SEND` does **not** block; it writes what fits and returns early.

`SYS_CHANNEL_SEND_ALL` does **not** block; it fails with `EAGAIN` if the ring
is full.  The caller is responsible for retrying (or using poll/wait to wait
for `POLLOUT` on the bridged VFS thing).

### Receiver blocks

`SYS_CHANNEL_RECV` parks the calling task in the channel's read wait queue until
data arrives or the write end is closed.

`SYS_CHANNEL_TRY_RECV` never blocks.

### `SYS_CHANNEL_WAIT`

Wait on one or more things simultaneously.  The caller supplies an array of
thing values and a flags word (`READABLE | WRITABLE`).  Returns the first
thing that becomes ready.  Blocks indefinitely until at least one thing is
ready.

---

## 6. Peer Death Semantics

### Write end closes (sender exits or calls `channel_close`)

1. Any threads blocked in `SYS_CHANNEL_RECV` are woken immediately.
2. `channel_recv` drains remaining buffered bytes normally.
3. After the buffer is empty, `channel_recv` returns `EPIPE` to signal
   end-of-stream.
4. Polling the read thing reports `POLLIN | POLLHUP`.

### Read end closes (receiver exits or calls `channel_close`)

1. Any threads blocked in `SYS_CHANNEL_RECV` are woken.
2. All further `SYS_CHANNEL_SEND` and `SYS_CHANNEL_SEND_ALL` calls return
   `EPIPE`.
3. Any threads blocked in `SYS_CHANNEL_SEND` are woken immediately.
4. Polling the write thing reports `POLLERR | POLLHUP`.

### Process crash / unexpected exit

The kernel closes all things owned by a process on exit, triggering the same
peer-death sequences above.  A service that holds a channel read thing will
observe `POLLERR | POLLHUP` on its write end within the same scheduling
quantum that the sender process exits.

---

## 7. Cleanup Semantics for Queued Handles

When a channel is closed while capability handles are still queued (i.e.
`channel_send_handle` was called but `channel_recv_handle` was not):

- The kernel drops the `Arc` reference it holds to each queued node.
- If no other references exist, the underlying VFS node is closed.
- No things are silently leaked into any process's thing table.

---

## 8. Error Reference

| Error | Condition |
|-------|-----------|
| `EBADF` | Thing value does not exist or has the wrong mode |
| `EINVAL` | `count == 0` or `count > 64` in `channel_wait`; `handles_count > 64` in `channel_send_msg` |
| `EAGAIN` | Ring is full (`send_all`) or empty (`try_recv`) |
| `EPIPE` | The peer endpoint is closed |
| `ENOMEM` | Thing table is full (`MAX_HANDLES = 1024`) |
| `EIO` | Internal ring write shorter than expected (provider send error) |

---

## 9. Bridging Channels to VFS Poll

`SYS_FD_FROM_HANDLE(thing) -> thing`

Wraps a channel thing in a VFS thing so it can participate in
`SYS_FS_POLL`.  The resulting thing inherits the source's mode (read or write).

Once bridged:
- `POLLIN` fires when the ring has bytes (read thing).
- `POLLOUT` fires when the ring has free space (write thing).
- `POLLHUP` fires when the peer has closed.
- `POLLERR` fires when the peer has closed the read end (write-thing perspective).

---

## 10. Diagnostics

The kernel maintains per-channel counters exposed under `/proc/ipc/channels`:

| Counter | Description |
|---------|-------------|
| `sends` | Total `channel_send` calls that wrote ≥1 byte |
| `recvs` | Total `channel_recv` calls that read ≥1 byte |
| `bytes_sent` | Cumulative bytes written |
| `bytes_recv` | Cumulative bytes read |
| `handles_sent` | Total capability things enqueued |
| `handles_recv` | Total capability things dequeued |
| `full_events` | Times `channel_send_all` returned `EAGAIN` due to full ring |
| `peer_deaths` | Times a peer closure was observed |

See `docs/concepts/ipc_diag.md` for how to read and interpret these counters.

---

## 11. See Also

- `kernel/src/ipc/port.rs` — ring buffer implementation
- `kernel/src/syscall/handlers/port.rs` — syscall handlers
- `abi/src/numbers.rs` — syscall numbers (`SYS_CHANNEL_*`)
- `stem/src/syscall/channel.rs` — userspace wrappers
- `docs/concepts/readiness.md` — poll/wait model
- `docs/concepts/ipc.md` — primitive overview and decision matrix

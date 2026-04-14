# IPC Health Diagnostics

This document explains how to inspect the health of kernel IPC — channels,
pipes, and VFS RPC — using the counters exposed through `/proc/ipc/`.

---

## Overview

The kernel maintains atomic 64-bit counters for every IPC hot path.  There is
**no locking** on the read path; values are written with `Relaxed` ordering so
they impose zero overhead on the fast path.

Counters are **cumulative** (they never reset while the kernel is running).
To compute rates (e.g. sends/sec) sample the file twice and divide by elapsed
time.

---

## Files under `/proc/ipc/`

| Path | Contents |
|------|----------|
| `/proc/ipc/channels` | Channel (port) send/receive counters |
| `/proc/ipc/pipes` | Anonymous pipe read/write counters |
| `/proc/ipc/vfs_rpc` | VFS provider RPC counters |

---

## `/proc/ipc/channels`

```
sends:         <n>
recvs:         <n>
bytes_sent:    <n>
bytes_recv:    <n>
handles_sent:  <n>
handles_recv:  <n>
full_events:   <n>
peer_deaths:   <n>
```

| Field | Meaning |
|-------|---------|
| `sends` | Total `channel_send` / `channel_send_all` calls that wrote ≥ 1 byte |
| `recvs` | Total `channel_recv` calls that read ≥ 1 byte |
| `bytes_sent` | Cumulative bytes written to channels |
| `bytes_recv` | Cumulative bytes read from channels |
| `handles_sent` | Capability handles enqueued via `channel_send_handle` / `channel_send_msg` |
| `handles_recv` | Capability handles dequeued via `channel_recv_handle` / `channel_recv_msg` |
| `full_events` | Times `channel_send_all` returned `EAGAIN` because the ring was full |
| `peer_deaths` | Times a send/recv observed that the peer had closed its end |

### Identifying hot channels

A high `sends` relative to `bytes_sent` suggests many small messages; consider
batching.  A rising `full_events` count means producers are outpacing consumers
— increase ring capacity or reduce send frequency.  A non-zero `peer_deaths`
count indicates unexpected process exits or handle leaks.

---

## `/proc/ipc/pipes`

```
writes:        <n>
reads:         <n>
bytes_written: <n>
bytes_read:    <n>
broken_pipe:   <n>
```

| Field | Meaning |
|-------|---------|
| `writes` | Total pipe write calls that wrote ≥ 1 byte |
| `reads` | Total pipe read calls that read ≥ 1 byte |
| `bytes_written` | Cumulative bytes written to pipes |
| `bytes_read` | Cumulative bytes read from pipes |
| `broken_pipe` | Times a write returned `EPIPE` because all readers had closed |

A rising `broken_pipe` count indicates that writers are outliving their
readers.  Check process lifecycle and pipe close ordering.

---

## `/proc/ipc/vfs_rpc`

```
requests:      <n>
errors:        <n>
dead_provider: <n>
```

| Field | Meaning |
|-------|---------|
| `requests` | Total VFS RPC requests attempted (including those that failed to send) |
| `errors` | Total requests that resulted in an error (includes dead provider) |
| `dead_provider` | Times a provider channel was found dead (send failure or EPIPE on response) |

Both `ProviderFs` (filesystem-level operations such as `lookup` and `rename`)
and `ProviderNode` (per-node operations such as `read`, `write`, `stat`) are
counted here.  `requests` is always ≥ `errors`; `dead_provider` is a subset
of `errors`.

A `dead_provider` count that grows monotonically usually means a provider
process crashed or was killed.  Restarting the provider (or the mount) will
stop the bleeding; the counter itself cannot be reset without rebooting.

---

## Practical recipes

### Sample sends/sec over a 5-second window

```sh
A=$(cat /proc/ipc/channels | grep '^sends:' | awk '{print $2}')
sleep 5
B=$(cat /proc/ipc/channels | grep '^sends:' | awk '{print $2}')
echo "sends/sec: $(( (B - A) / 5 ))"
```

### Check for queue pressure

```sh
cat /proc/ipc/channels | grep full_events
```

A non-zero and growing value means at least one channel ring is saturated.

### Check for dead VFS providers

```sh
cat /proc/ipc/vfs_rpc | grep dead_provider
```

### Watch all IPC counters continuously

```sh
while true; do
    echo "=== $(date) ==="
    echo "--- channels ---"
    cat /proc/ipc/channels
    echo "--- pipes ---"
    cat /proc/ipc/pipes
    echo "--- vfs_rpc ---"
    cat /proc/ipc/vfs_rpc
    sleep 2
done
```

---

## Implementation notes

Counters live in `kernel/src/ipc/diag.rs` as `static AtomicU64` values.
They are incremented at the following sites:

| Counter | Increment site |
|---------|----------------|
| `CHANNEL_SENDS` / `CHANNEL_BYTES_SENT` | `sys_channel_send`, `sys_channel_send_all` |
| `CHANNEL_RECVS` / `CHANNEL_BYTES_RECV` | `sys_channel_recv`, `sys_channel_recv_blocking` |
| `CHANNEL_HANDLES_SENT` | `sys_channel_send_handle`, `sys_channel_send_msg` |
| `CHANNEL_HANDLES_RECV` | `sys_channel_recv_handle`, `sys_channel_recv_msg` |
| `CHANNEL_FULL_EVENTS` | `sys_channel_send_all` when ring is full |
| `CHANNEL_PEER_DEATHS` | send/recv when peer has closed |
| `PIPE_WRITES` / `PIPE_BYTES_WRITTEN` | `ipc::pipe::write` |
| `PIPE_READS` / `PIPE_BYTES_READ` | `ipc::pipe::read` |
| `PIPE_BROKEN_PIPE` | `ipc::pipe::write` when no readers |
| `VFS_RPC_REQUESTS` | `ProviderChannel::rpc`, `ProviderChannelRef::rpc` — incremented before send |
| `VFS_RPC_ERRORS` | same, on write failure or EPIPE on response |
| `VFS_RPC_DEAD_PROVIDER` | same, whenever the provider port is unreachable (always paired with `VFS_RPC_ERRORS` via `record_dead_provider_error()`) |

---

## See also

- `kernel/src/ipc/diag.rs` — counter definitions and text renderers
- `kernel/src/vfs/procfs.rs` — `/proc/ipc/` VFS nodes
- `docs/concepts/channel_semantics.md` — channel semantics and lifecycle
- `docs/concepts/vfs_rpc_provider.md` — VFS provider protocol

# Waitable Network Events (VFS/Kernel Contract)

## Problem

Blocking network operations in userspace (`std::net` and PAL wrappers) previously used repeated `read/write` retries with short sleeps on `EAGAIN`.

This caused:
- avoidable CPU burn while idle
- poor scalability with many blocked sockets
- latency/throughput tuning tradeoffs from fixed polling intervals

## Kernel/VFS Primitive Used

Thing-OS already exposes a blocking readiness primitive via `SYS_FS_POLL`:
- syscall: `SYS_FS_POLL` (`0x400B`)
- ABI entry: `abi::syscall::PollFd`
- flags: `abi::syscall::poll_flags::{POLLIN, POLLOUT, POLLERR, POLLHUP, POLLNVAL}`

Kernel implementation (`kernel/src/syscall/handlers/vfs.rs`) already provides:
- immediate probe of all FDs
- waiter registration with no-lost-wakeup re-probe
- blocking task park/unpark
- timeout handling
- interrupt handling (`EINTR`)

## Network Blocking Policy

When socket/data/accept/DNS fds return `EAGAIN` in blocking mode:
1. Block in kernel with `SYS_FS_POLL` for the relevant readiness bit.
2. Retry operation after wake.
3. Preserve timeout deadlines (`ETIMEDOUT`) and nonblocking behavior (`EAGAIN`).
4. If `SYS_FS_POLL` is not available, fall back to sleep-based retry.

Readiness mapping:
- `accept` / `read` / `recv_from` / DNS reply reads: `POLLIN`
- `write` / `write_vectored`: `POLLOUT`

## Scope Implemented

- Thing-OS std PAL net backend switched to `SYS_FS_POLL` waits in blocking retry paths.
- stem PAL net uses `vfs_poll` in accept/connect/hostname-resolution loops.
- Fallback path retained for older kernels lacking poll support.

## std::net Semantics

Preserved semantics:
- nonblocking sockets still return `EAGAIN` immediately
- timeout-based operations still return `ETIMEDOUT`
- peer-close detection paths still map to expected IO errors

## Benchmark Plan

### Goals

Demonstrate:
- lower idle CPU while blocked on accepts/reads
- improved concurrency scaling under many waiting sockets

### Workloads

1. Idle accept:
- one listener blocked in `accept()` with no clients for 60s

2. Idle read:
- N connected clients blocked in `read()` with no payload for 60s

3. Concurrency scaling:
- increasing blocked connection counts (`N = 32, 128, 512, 1024`)
- measure scheduler idle ratio / CPU usage and wake latency

### Metrics

- average CPU utilization while blocked
- wake-to-read latency p50/p95/p99
- throughput under active load at each N

### Pass Criteria

- idle CPU lower than prior sleep-retry baseline
- no regression in timeout/nonblocking behavior
- improved or stable throughput as N increases

## Notes

`vendor/rust/` is intentionally not tracked by this repository. Changes to std internals must be committed to the rust fork repository (`dancxjo/rust-thingos`).

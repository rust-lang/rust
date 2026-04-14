# Waitables

Thing-OS exposes a unified wait fabric for all asynchronous readiness events: file
descriptors, ports, task exit, timers, and IRQ delivery. This document describes
the waitable handle shapes and their semantics.

## Model

There are three public waitable handle shapes:

- `FsWatch`: a persistent stream handle for filesystem/device change notifications.
- `OpHandle`: a one-shot completion handle for async operations (e.g., async writes).
- `ReadyCondition`: a one-shot condition helper that arms against a watch and
  re-checks state on each wake.

Internally these map onto the kernel's poll/waiter infrastructure, but userland code
does not need to name internal opcodes or reply-cell plumbing directly.

## Watch Versus Condition

`FsWatch` and `ReadyCondition` are intentionally different.

- A watch is stream semantics. It stays active until closed, becomes readable while
  unread event payloads exist, and may report overflow.
- A condition is one-shot semantics. It answers "has this resource become ready yet?"
  and disarms after success.

The first implementation of `ReadyCondition` is deliberately simple:

- it performs an immediate readiness check when armed
- if the condition is already satisfied, the caller does not need to enter `wait_many`
- otherwise it waits on an underlying `FsWatch`
- when the watch wakes, it drains pending payloads and re-evaluates the condition
  against current resource state

This keeps condition waits declarative without pretending they are the same thing as
event streams.

## Operation Handles

Async operations produce waitable completion handles.

- `OpHandle` is valid input to `wait_many` through `WaitKind::Op`
- readiness is reported with `ready::DONE`
- failures are reported with `ready::DONE | ready::ERROR`
- the wait result carries the success value or errno payload

Lifecycle rules are explicit:

- pending handles remain registered until completion or cancellation
- `take_result()` consumes the completion and drops the kernel handle
- dropping or canceling a handle wakes any blocked waiters, which then observe
  `ENOENT` on the next poll

This makes destruction visible instead of silently sleeping forever.

## Overflow And Backpressure

File watches are persistent and bounded.

- the kernel keeps a bounded event history per watch
- if a watch falls behind the retained history, it reports overflow
- overflow is sticky until the consumer observes it
- condition waits do not queue condition hits; they re-check readiness after each
  wake and after overflow

This means stream consumers can detect loss explicitly, while condition consumers
avoid flooding because they only care whether the resource is ready now.

## Public API Direction

The canonical readiness API for userland is FD-centric:

- For polling one or more FDs for I/O readiness, use **`SYS_FS_POLL`** (stem
  wrapper: `vfs_poll`).  Pipes, sockets, and channel ends bridged via
  `SYS_FS_FD_FROM_HANDLE` all use the same `PollFd` interface.
- To block on a mix of FDs, ports, task exit, and IRQs in a single call, use
  **`SYS_WAIT_MANY`** with `WaitKind::Fd` (stem wrapper: `WaitSet::add_fd_readable`
  / `add_fd_writable`).
- Higher-level helpers: `FsWatch` for change-notification streams, `OpHandle`
  for one-shot async completions, `ReadyCondition` for declarative condition
  waits.

New userland code must use VFS-oriented handle types.  Do not introduce new
APIs that reference graph watches, graph ops, or graph-era naming.

### Deprecated `WaitKind` values

`WaitKind::GraphOp` and `WaitKind::RootWatch` are **deprecated** and return
`ENOSYS`.  The stem `WaitSet::add_graph_op` method is deprecated accordingly.
Use `add_fd_readable` / `add_fd_writable` for all FD-based readiness.

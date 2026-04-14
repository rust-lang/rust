# `wait_many` design note

`wait_many` is the first unified kernel wait primitive for heterogeneous readiness. The current implementation is single-shot rather than a persistent waitset object, but its internals are structured around the same three phases a future waitset will need: poll, register, and unregister.

## Readiness semantics

- Ports are level-triggered. A read wait stays ready while unread bytes remain. A write wait stays ready while buffer space remains.
- Root watches are level-triggered. A watch is ready when it has unread stream data to drain or when it needs to report overflow.
- Graph operation handles are one-shot. An op becomes ready when its completion status is known.
- Task exit is level-triggered. Once a task is dead, its exit status remains observable.
- IRQ readiness is based on pending interrupt counts for an already-subscribed vector.
- Timeout is supplied as the syscall timeout parameter and is returned as a synthetic ready result with `WaitKind::Timeout`.

## Lifetime and error semantics

- Closed or invalid ports surface as ready results with the `ERROR` or `HANGUP` bits rather than sleeping forever.
- Destroyed or stale watches wake blocked waiters and are reported as `ERROR`.
- Dropped graph operation handles wake blocked waiters and are reported as `ERROR`.
- Missing task targets are reported as `ERROR` with `ECHILD`.
- Timeouts do not raise a syscall error; they return a timeout result so callers can keep one event loop shape.

## Race handling

The syscall follows the standard no-lost-wakeup pattern:

1. Poll all specs for immediate readiness.
2. If nothing is ready, register the current task with every waitable source.
3. Re-poll all specs after registration.
4. Only block if the second poll still finds nothing ready.

This works with the scheduler’s existing `wake_pending` path, so a wake that lands between registration and the actual block does not get lost.

## Path to persistent waitsets

A persistent waitset object can reuse the same internal contracts:

- poll a waitable for its current state
- register/unregister a waiter token against a waitable
- gather all ready results into a bounded result buffer

The main additional work would be moving the spec list and registrations into a kernel object so callers can mutate membership incrementally instead of resubmitting the full spec array each call.

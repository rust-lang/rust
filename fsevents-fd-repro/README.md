# fsevents-fd-repro

Standalone reproducer attempt for rust-lang/rust#124105, no notify dependency.
Requires macOS.

    cargo run

## Two hypotheses, not one

The CI evidence says only *where* the abort happened, not *why*. Two stories fit:

- **A.** FSEvents, or CoreFoundation's CFURL resolution, closes a descriptor it
  does not own, and std is the victim.
- **B.** std's own `remove_dir_all` mishandles a descriptor on a large tree, and
  FSEvents is incidental.

So this runs three worker kinds concurrently instead of assuming A:

- **faithful** mirrors notify's failing test: a fresh 8194-directory tree, a
  per-path CFURL file-reference round trip, a stream created with
  `FileEvents|NoDefer|WatchRoot`, a new thread per stream that creates a run
  loop, schedules, watches `FSEventStreamStart` fail, invalidates, releases and
  exits, then `remove_dir_all` of *that same tree*.
- **live** a stream that actually starts, is serviced by a running run loop, and
  has its tree deleted underneath it.
- **control** the same tree churn with no FSEvents in the process at all.

**If only `control` trips, hypothesis B is right and FSEvents is a red herring.**
That is the most useful single result this program can produce, and it costs
almost nothing to run.

## How detection works

It does not rely on std's checks firing. It holds a pool of canary descriptors
and, for each one:

- probes with `fcntl(F_GETFD)` rather than `fstat`. std's own comment on
  `debug_assert_fd_is_open` notes that EBADF from ordinary IO syscalls can be
  bubbled up from a FUSE server with the descriptor perfectly valid, so `fstat`
  EBADF alone is a false-positive path. `F_GETFD` queries the process descriptor
  table and cannot be faked that way.
- compares `(st_dev, st_ino)` against what the canary was opened on. This is the
  important one: `open` returns the *lowest* free descriptor, so a stray close
  frees a number that is re-handed-out within microseconds, long before any
  poll. Checking liveness alone would miss essentially every occurrence.
  Checking identity catches it after the fact, because a reused descriptor
  points at a different inode.

The pool is also rotated, so canaries keep landing on freshly recycled
descriptor numbers rather than sitting in a fixed low block that nothing ever
reuses.

## Knobs

`PATHS` (4097), `CANARIES` (64), `SECONDS` (300), `FAITHFUL` (1), `LIVE` (1),
`CONTROL` (1), `CFURL` (1, set 0 to skip the file-reference round trip).

    SECONDS=1800 FAITHFUL=2 CONTROL=2 cargo run

To test hypothesis B on its own: `FAITHFUL=0 LIVE=0 CONTROL=4`.

## Reading the output

`start_fail` should track `cycles`. If it stays at 0 the faithful worker is not
exercising its path at all, either because `PATHS` is under the cap or because
the FFI is wrong; the program warns about this a few seconds in rather than
after the full run. `create_null` above 0 means those cycles did nothing.

A worker panicking (for instance with `closedir: Bad file descriptor`) is
detected via a panic hook and reported at exit, rather than being swallowed by
the thread and reported as a clean run.

## Status

UNVERIFIED. Written on Linux, so it has only ever been type checked and linted
for `x86_64-apple-darwin`, never built or run. Expect problems on first contact
with a Mac. It is a race, so a clean run is weak evidence either way; the
`control`-only result is the one that would actually settle something.

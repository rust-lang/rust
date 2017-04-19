# `compiler_fences`

The tracking issue for this feature is: [#41091]

[#41091]: https://github.com/rust-lang/rust/issues/41091

------------------------

The `compiler_fences` feature exposes the `compiler_fence` function
in `std::sync::atomic`. This function is conceptually similar to C++'s
`atomic_signal_fence`, which can currently only be accessed in nightly
Rust using the `atomic_singlethreadfence_*` instrinsic functions in
`core`, or through the mostly equivalent literal assembly:

```rust
#![feature(asm)]
unsafe { asm!("" ::: "memory" : "volatile") };
```

A `compiler_fence` restricts the kinds of memory re-ordering the
compiler is allowed to do. Specifically, depending on the given ordering
semantics, the compiler may be disallowed from moving reads or writes
from before or after the call to the other side of the call to
`compiler_fence`. Note that it does **not** prevent the *hardware*
from doing such re-ordering. This is not a problem in a single-threaded,
execution context, but when other threads may modify memory at the same
time, stronger synchronization primitives are required.

## Examples

`compiler_fence` is generally only useful for preventing a thread from
racing *with itself*. That is, if a given thread is executing one piece
of code, and is then interrupted, and starts executing code elsewhere
(while still in the same thread, and conceptually still on the same
core). In traditional programs, this can only occur when a signal
handler is registered. In more low-level code, such situations can also
arise when handling interrupts, when implementing green threads with
pre-emption, etc.

To give a straightforward example of when a `compiler_fence` is
necessary, consider the following example:

```rust
# use std::sync::atomic::{AtomicBool, AtomicUsize};
# use std::sync::atomic::{ATOMIC_BOOL_INIT, ATOMIC_USIZE_INIT};
# use std::sync::atomic::Ordering;
static IMPORTANT_VARIABLE: AtomicUsize = ATOMIC_USIZE_INIT;
static IS_READY: AtomicBool = ATOMIC_BOOL_INIT;

fn main() {
    IMPORTANT_VARIABLE.store(42, Ordering::Relaxed);
    IS_READY.store(true, Ordering::Relaxed);
}

fn signal_handler() {
    if IS_READY.load(Ordering::Relaxed) {
        assert_eq!(IMPORTANT_VARIABLE.load(Ordering::Relaxed), 42);
    }
}
```

The way it is currently written, the `assert_eq!` is *not* guaranteed to
succeed, despite everything happening in a single thread. To see why,
remember that the compiler is free to swap the stores to
`IMPORTANT_VARIABLE` and `IS_READ` since they are both
`Ordering::Relaxed`. If it does, and the signal handler is invoked right
after `IS_READY` is updated, then the signal handler will see
`IS_READY=1`, but `IMPORTANT_VARIABLE=0`.

Using a `compiler_fence`, we can remedy this situation:

```rust
#![feature(compiler_fences)]
# use std::sync::atomic::{AtomicBool, AtomicUsize};
# use std::sync::atomic::{ATOMIC_BOOL_INIT, ATOMIC_USIZE_INIT};
# use std::sync::atomic::Ordering;
use std::sync::atomic::compiler_fence;

static IMPORTANT_VARIABLE: AtomicUsize = ATOMIC_USIZE_INIT;
static IS_READY: AtomicBool = ATOMIC_BOOL_INIT;

fn main() {
    IMPORTANT_VARIABLE.store(42, Ordering::Relaxed);
    // prevent earlier writes from being moved beyond this point
    compiler_fence(Ordering::Release);
    IS_READY.store(true, Ordering::Relaxed);
}

fn signal_handler() {
    if IS_READY.load(Ordering::Relaxed) {
        assert_eq!(IMPORTANT_VARIABLE.load(Ordering::Relaxed), 42);
    }
}
```

A deeper discussion of compiler barriers with various re-ordering
semantics (such as `Ordering::SeqCst`) is beyond the scope of this text.
Curious readers are encouraged to read the Linux kernel's discussion of
[memory barriers][1], the C++ references on [`std::memory_order`][2] and
[`atomic_signal_fence`][3], and [this StackOverflow answer][4] for
further details.

[1]: https://www.kernel.org/doc/Documentation/memory-barriers.txt
[2]: http://en.cppreference.com/w/cpp/atomic/memory_order
[3]: http://www.cplusplus.com/reference/atomic/atomic_signal_fence/
[4]: http://stackoverflow.com/a/18454971/472927

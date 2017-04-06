# `compiler_barriers`

The tracking issue for this feature is: [#41092]

[#41092]: https://github.com/rust-lang/rust/issues/41092

------------------------

The `compiler_barriers` feature exposes the `compiler_barrier` function
in `std::sync::atomic`. This function is conceptually similar to C++'s
`atomic_signal_fence`, which can currently only be accessed in nightly
Rust using the `atomic_singlethreadfence_*` instrinsic functions in
`core`, or through the mostly equivalent literal assembly:

```rust
#![feature(asm)]
unsafe { asm!("" ::: "memory" : "volatile") };
```

A `compiler_barrier` restricts the kinds of memory re-ordering the
compiler is allowed to do. Specifically, depending on the given ordering
semantics, the compiler may be disallowed from moving reads or writes
from before or after the call to the other side of the call to
`compiler_barrier`.

## Examples

The need to prevent re-ordering of reads and writes often arises when
working with low-level devices. Consider a piece of code that interacts
with an ethernet card with a set of internal registers that are accessed
through an address port register (`a: &mut usize`) and a data port
register (`d: &usize`). To read internal register 5, the following code
might then be used:

```rust
fn read_fifth(a: &mut usize, d: &usize) -> usize {
    *a = 5;
    *d
}
```

In this case, the compiler is free to re-order these two statements if
it thinks doing so might result in better performance, register use, or
anything else compilers care about. However, in doing so, it would break
the code, as `x` would be set to the value of some other device
register!

By inserting a compiler barrier, we can force the compiler to not
re-arrange these two statements, making the code function correctly
again:

```rust
#![feature(compiler_barriers)]
use std::sync::atomic;

fn read_fifth(a: &mut usize, d: &usize) -> usize {
    *a = 5;
    atomic::compiler_barrier(atomic::Ordering::SeqCst);
    *d
}
```

Compiler barriers are also useful in code that implements low-level
synchronization primitives. Consider a structure with two different
atomic variables, with a dependency chain between them:

```rust
use std::sync::atomic;

fn thread1(x: &atomic::AtomicUsize, y: &atomic::AtomicUsize) {
    x.store(1, atomic::Ordering::Release);
    let v1 = y.load(atomic::Ordering::Acquire);
}
fn thread2(x: &atomic::AtomicUsize, y: &atomic::AtomicUsize) {
    y.store(1, atomic::Ordering::Release);
    let v2 = x.load(atomic::Ordering::Acquire);
}
```

This code will guarantee that `thread1` sees any writes to `y` made by
`thread2`, and that `thread2` sees any writes to `x`. Intuitively, one
might also expect that if `thread2` sees `v2 == 0`, `thread1` must see
`v1 == 1` (since `thread2`'s store happened before its `load`, and its
load did not see `thread1`'s store). However, the code as written does
*not* guarantee this, because the compiler is allowed to re-order the
store and load within each thread. To enforce this particular behavior,
a call to `compiler_barrier(Ordering::SeqCst)` would need to be inserted
between the `store` and `load` in both functions.

Compiler barriers with weaker re-ordering semantics (such as
`Ordering::Acquire`) can also be useful, but are beyond the scope of
this text. Curious readers are encouraged to read the Linux kernel's
discussion of [memory barriers][1], as well as C++ references on
[`std::memory_order`][2] and [`atomic_signal_fence`][3].

[1]: https://www.kernel.org/doc/Documentation/memory-barriers.txt
[2]: http://en.cppreference.com/w/cpp/atomic/memory_order
[3]: http://www.cplusplus.com/reference/atomic/atomic_signal_fence/

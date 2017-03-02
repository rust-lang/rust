q- Feature Name: atomic_access
- Start Date: 2016-06-15
- RFC PR: [rust-lang/rfcs#1649](https://github.com/rust-lang/rfcs/pull/1649)
- Rust Issue: [rust-lang/rust#35603](https://github.com/rust-lang/rust/issues/35603)

# Summary
[summary]: #summary

This RFC adds the following methods to atomic types:

```rust
impl AtomicT {
    fn get_mut(&mut self) -> &mut T;
    fn into_inner(self) -> T;
}
```

It also specifies that the layout of an `AtomicT` type is always the same as the underlying `T` type. So, for example, `AtomicI32` is guaranteed to be transmutable to and from `i32`.

# Motivation
[motivation]: #motivation

## `get_mut` and `into_inner`

These methods are useful for accessing the value inside an atomic object directly when there are no other threads accessing it. This is guaranteed by the mutable reference and the move, since it means there can be no other live references to the atomic.

A normal load/store is different from a `load(Relaxed)` or `store(Relaxed)` because it has much weaker synchronization guarantees, which means that the compiler can produce more efficient code. In particular, LLVM currently treats all atomic operations (even relaxed ones) as volatile operations, which means that it does not perform any optimizations on them. For example, it will not eliminate a `load(Relaxed)` even if the results of the load is not used anywhere.

`get_mut` in particular is expected to be useful in `Drop` implementations where you have a `&mut self` and need to read the value of an atomic. `into_inner` somewhat overlaps in functionality with `get_mut`, but it is included to allow extracting the value without requiring the atomic object to be mutable. These methods mirror `Mutex::get_mut` and `Mutex::into_inner`.

## Atomic type layout

The layout guarantee is mainly intended to be used for FFI, where a variable of a non-atomic type needs to be modified atomically. The most common example of this is the Linux `futex` system call which takes an `int*` parameter pointing to an integer that is atomically modified by both userspace and the kernel.

Rust code invoking the `futex` system call so far has simply passed the address of the atomic object directly to the system call. However this makes the assumption that the atomic type has the same layout as the underlying integer type, which is not currently guaranteed by the documentation.

This also allows the reverse operation by casting a pointer: it allows Rust code to atomically modify a value that was not declared as a atomic type. This is useful when dealing with FFI structs that are shared with a thread managed by a C library. Another example would be to atomically modify a value in a memory mapped file that is shared with another process.

# Detailed design
[design]: #detailed-design

The actual implementations of these functions are mostly trivial since they are based on `UnsafeCell::get`.

The existing implementations of atomic types already have the same layout as the underlying types (even `AtomicBool` and `bool`), so no change is needed here apart from the documentation.

# Drawbacks
[drawbacks]: #drawbacks

The functionality of `into_inner` somewhat overlaps with `get_mut`.

We lose the ability to change the layout of atomic types, but this shouldn't be necessary since these types map directly to hardware primitives.

# Alternatives
[alternatives]: #alternatives

The functionality of `get_mut` and `into_inner` can be implemented using `load(Relaxed)`, however the latter can result in worse code because it is poorly handled by the optimizer.

# Unresolved questions
[unresolved]: #unresolved-questions

None

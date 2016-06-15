- Feature Name: atomic_access
- Start Date: 2016-06-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add the following methods to atomic types:

```rust
impl AtomicT {
    fn get_mut(&mut self) -> &mut T;
    fn into_inner(self) -> T;
    fn as_raw(&self) -> *mut T;
    unsafe fn from_raw(ptr: *mut T) -> &AtomicT;
}
```

# Motivation
[motivation]: #motivation

## `get_mut` and `into_inner`

These methods are useful for accessing the value inside an atomic object directly when there are no other threads accessing it. This is guaranteed by the mutable reference and the move, since it means there can be no other live references to the atomic.

A normal load/store is different from a `load(Relaxed)` or `store(Relaxed)` because it has much weaker synchronization guarantees, which means that the compiler can produce more efficient code. In particular, LLVM currently treats all atomic operations (even relaxed ones) as volatile operations, which means that it does not perform any optimizations on them. For example, it will not eliminate a `load(Relaxed)` even if the results of the load is not used anywhere.

`get_mut` in particular is expected to be useful in `Drop` implementations where you have a `&mut self` and need to read the value of an atomic. `into_inner` somewhat overlaps in functionality with `get_mut`, but it is included to allow extracting the value without requiring the atomic object to be mutable. These methods mirror `Mutex::get_mut` and `Mutex::into_inner`.

## `as_raw` and `from_raw`

These methods are mainly intended to be used for FFI, where a variable of a non-atomic type needs to be modified atomically. The most common example of this is the Linux `futex` system call which takes an `int*` parameter pointing to an integer that is atomically modified by both userspace and the kernel.

Rust code invoking the `futex` system call so far has simply passed the address of the atomic object directly to the system call. However this makes the assumption that the atomic type has the same layout as the underlying integer type. Using `as_raw` instead makes it clear that the resulting pointer will point to the integer value inside the atomic object.

`from_raw` provides the reverse operation: it allows Rust code to atomically modify a value that was not declared as a atomic type. This is useful when dealing with FFI structs that are shared with a thread managed by a C library. Another example would be to atomically modify a value in a memory mapped file that is shared with another process.

# Detailed design
[design]: #detailed-design

The actual implementations of these functions are mostly trivial since they are based on `UnsafeCell::get`. The only exception is `from_raw` which will cast the given pointer to a different type, but that should also be fine.

# Drawbacks
[drawbacks]: #drawbacks

The functionality of `into_inner` somewhat overlaps with `get_mut`.

`from_raw` returns an unbounded lifetime.

# Alternatives
[alternatives]: #alternatives

The functionality of `get_mut` and `into_inner` can be implemented using `load(Relaxed)`, however the latter can result in worse code because it is poorly handled by the optimizer.

The functionality of `as_raw` and `from_raw` could be achieved using transmutes instead, however this requires making assumptions about the internal layout of the atomic types.

# Unresolved questions
[unresolved]: #unresolved-questions

None

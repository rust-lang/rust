- Feature Name: `extended_compare_and_swap`
- Start Date: 2016-1-5
- RFC PR: [rust-lang/rfcs#1443](https://github.com/rust-lang/rfcs/pull/1443)
- Rust Issue: [rust-lang/rust#31767](https://github.com/rust-lang/rust/issues/31767)

# Summary
[summary]: #summary

Rust currently provides a `compare_and_swap` method on atomic types, but this method only exposes a subset of the functionality of the C++11 equivalents [`compare_exchange_strong` and `compare_exchange_weak`](http://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange):

- `compare_and_swap` maps to the C++11 `compare_exchange_strong`, but there is no Rust equivalent for `compare_exchange_weak`. The latter is allowed to fail spuriously even when the comparison succeeds, which allows the compiler to generate better assembly code when the compare and swap is used in a loop.

- `compare_and_swap` only has a single memory ordering parameter, whereas the C++11 versions have two: the first describes the memory ordering when the operation succeeds while the second one describes the memory ordering on failure.

# Motivation
[motivation]: #motivation

While all of these variants are identical on x86, they can allow more efficient code to be generated on architectures such as ARM:

- On ARM, the strong variant of compare and swap is compiled into an `LDREX` / `STREX` loop which restarts the compare and swap when a spurious failure is detected. This is unnecessary for many lock-free algorithms since the compare and swap is usually already inside a loop and a spurious failure is often caused by another thread modifying the atomic concurrently, which will probably cause the compare and swap to fail anyways.

- When Rust lowers `compare_and_swap` to LLVM, it uses the same memory ordering type for success and failure, which on ARM adds extra memory barrier instructions to the failure path. Most lock-free algorithms which make use of compare and swap in a loop only need relaxed ordering on failure since the operation is going to be restarted anyways.

# Detailed design
[design]: #detailed-design

Since `compare_and_swap` is stable, we can't simply add a second memory ordering parameter to it. This RFC proposes deprecating the `compare_and_swap` function and replacing it with `compare_exchange` and `compare_exchange_weak`, which match the names of the equivalent C++11 functions (with the `_strong` suffix removed).

## `compare_exchange`

A new method is instead added to atomic types:

```rust
fn compare_exchange(&self, current: T, new: T, success: Ordering, failure: Ordering) -> T;
```

The restrictions on the failure ordering are the same as C++11: only `SeqCst`, `Acquire` and `Relaxed` are allowed and it must be equal or weaker than the success ordering. Passing an invalid memory ordering will result in a panic, although this can often be optimized away since the ordering is usually statically known.

The documentation for the original `compare_and_swap` is updated to say that it is equivalent to `compare_exchange` with the following mapping for memory orders:

Original | Success | Failure
-------- | ------- | -------
Relaxed  | Relaxed | Relaxed
Acquire  | Acquire | Acquire
Release  | Release | Relaxed
AcqRel   | AcqRel  | Acquire
SeqCst   | SeqCst  | SeqCst

## `compare_exchange_weak`

A new method is instead added to atomic types:

```rust
fn compare_exchange_weak(&self, current: T, new: T, success: Ordering, failure: Ordering) -> (T, bool);
```

`compare_exchange` does not need to return a success flag because it can be inferred by checking if the returned value is equal to the expected one. This is not possible for `compare_exchange_weak` because it is allowed to fail spuriously, which means that it could fail to perform the swap even though the returned value is equal to the expected one.

A lock free algorithm using a loop would use the returned bool to determine whether to break out of the loop, and if not, use the returned value for the next iteration of the loop.

## Intrinsics

These are the existing intrinsics used to implement `compare_and_swap`:

```rust
    pub fn atomic_cxchg<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acq<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_rel<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acqrel<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_relaxed<T>(dst: *mut T, old: T, src: T) -> T;
```

The following intrinsics need to be added to support relaxed memory orderings on failure:

```rust
    pub fn atomic_cxchg_acqrel_failrelaxed<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_failacq<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_failrelaxed<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acq_failrelaxed<T>(dst: *mut T, old: T, src: T) -> T;
```

The following intrinsics need to be added to support `compare_exchange_weak`:

```rust
    pub fn atomic_cxchg_weak<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_acq<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_rel<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_acqrel<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_relaxed<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_acqrel_failrelaxed<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_failacq<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_failrelaxed<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    pub fn atomic_cxchg_weak_acq_failrelaxed<T>(dst: *mut T, old: T, src: T) -> (T, bool);
```

# Drawbacks
[drawbacks]: #drawbacks

Ideally support for failure memory ordering would be added by simply adding an extra parameter to the existing `compare_and_swap` function. However this is not possible because `compare_and_swap` is stable.

This RFC proposes deprecating a stable function, which may not be desirable.

# Alternatives
[alternatives]: #alternatives

One alternative for supporting failure orderings is to add new enum variants to `Ordering` instead of adding new methods with two ordering parameters. The following variants would need to be added: `AcquireFailRelaxed`, `AcqRelFailRelaxed`, `SeqCstFailRelaxed`, `SeqCstFailAcquire`. The downside is that the names are quite ugly and are only valid for `compare_and_swap`, not other atomic operations. It is also a breaking change to a stable enum.

Another alternative is to not deprecate `compare_and_swap` and instead add `compare_and_swap_explicit`, `compare_and_swap_weak` and `compare_and_swap_weak_explicit`. However the distiniction between the explicit and non-explicit isn't very clear and can lead to some confusion.

Not doing anything is also a possible option, but this will cause Rust to generate worse code for some lock-free algorithms.

# Unresolved questions
[unresolved]: #unresolved-questions

None

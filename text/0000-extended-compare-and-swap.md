- Feature Name: extended_compare_and_swap
- Start Date: 2016-1-5
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

## Memory ordering on failure

Since `compare_and_swap` is stable, we can't simply add a second memory ordering parameter to it. A new method is instead added to atomic types:

```rust
fn compare_and_swap_explicit(&self, current: T, new: T, success: Ordering, failure: Ordering) -> T;
```

The restrictions on the failure ordering are the same as C++11: only `SeqCst`, `Acquire` and `Relaxed` are allowed and it must be equal or weaker than the success ordering.

The documentation for the original `compare_and_swap` is updated to say that it is equivalent to `compare_and_swap_explicit` with the following mapping for memory orders:

Original | Success | Failure
-------- | ------- | -------
Relaxed  | Relaxed | Relaxed
Acquire  | Acquire | Acquire
Release  | Release | Relaxed
AcqRel   | AcqRel  | Acquire
SeqCst   | SeqCst  | SeqCst

## `compare_and_swap_weak`

Two new methods are added to atomic types:

```rust
fn compare_and_swap_weak(&self, current: T, new: T, order: Ordering) -> (T, bool);
fn compare_and_swap_weak_explicit(&self, current: T, new: T, success: Ordering, failure: Ordering) -> (T, bool);
```

`compare_and_swap` does not need to return a success flag because it can be inferred by checking if the returned value is equal to the expected one. This is not possible for `compare_and_swap_weak` because it is allowed to fail spuriously, which means that it could fail to perform the swap even though the returned value is equal to the expected one.

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

The following intrinsics need to be added to support `compare_and_swap_weak`:

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

For consistency with `compare_and_swap`, `compare_and_swap_weak` also has a separate explicit variant with two memory ordering parameters, even though ideally only a single method would be required.

# Alternatives
[alternatives]: #alternatives

One alternative for supporting failure orderings is to add new enum variants to `Ordering` instead of adding new methods with two ordering parameters. The following variants would need to be added: `AcquireFailRelaxed`, `AcqRelFailRelaxed`, `SeqCstFailRelaxed`, `SeqCstFailAcquire`. The downside is that the names are quite ugly and are only valid for `compare_and_swap`, not other atomic operations.

Not doing anything is also a possible option, but this will cause Rust to generate worse code for some lock-free algorithms.

# Unresolved questions
[unresolved]: #unresolved-questions

None

- Feature Name: `integer_atomics`
- Start Date: 2016-03-14
- RFC PR: [rust-lang/rfcs#1543](https://github.com/rust-lang/rfcs/pull/1543)
- Rust Issue: [rust-lang/rust#32976](https://github.com/rust-lang/rust/issues/32976)

# Summary
[summary]: #summary

This RFC basically changes `core::sync::atomic` to look like this:

```rust
#[cfg(target_has_atomic = "8")]
struct AtomicBool {}
#[cfg(target_has_atomic = "8")]
struct AtomicI8 {}
#[cfg(target_has_atomic = "8")]
struct AtomicU8 {}
#[cfg(target_has_atomic = "16")]
struct AtomicI16 {}
#[cfg(target_has_atomic = "16")]
struct AtomicU16 {}
#[cfg(target_has_atomic = "32")]
struct AtomicI32 {}
#[cfg(target_has_atomic = "32")]
struct AtomicU32 {}
#[cfg(target_has_atomic = "64")]
struct AtomicI64 {}
#[cfg(target_has_atomic = "64")]
struct AtomicU64 {}
#[cfg(target_has_atomic = "128")]
struct AtomicI128 {}
#[cfg(target_has_atomic = "128")]
struct AtomicU128 {}
#[cfg(target_has_atomic = "ptr")]
struct AtomicIsize {}
#[cfg(target_has_atomic = "ptr")]
struct AtomicUsize {}
#[cfg(target_has_atomic = "ptr")]
struct AtomicPtr<T> {}
```

# Motivation
[motivation]: #motivation

Many lock-free algorithms require a two-value `compare_exchange`, which is effectively twice the size of a `usize`. This would be implemented by atomically swapping a struct containing two members.

Another use case is to support Linux's futex API. This API is based on atomic `i32` variables, which currently aren't available on x86_64 because `AtomicIsize` is 64-bit.

# Detailed design
[design]: #detailed-design

## New atomic types

The `AtomicI8`, `AtomicI16`, `AtomicI32`, `AtomicI64` and `AtomicI128` types are added along with their matching `AtomicU*` type. These have the same API as the existing `AtomicIsize` and `AtomicUsize` types. Note that support for 128-bit atomics is dependent on the [i128/u128 RFC](https://github.com/rust-lang/rfcs/pull/1504) being accepted.

## Target support

One problem is that it is hard for a user to determine if a certain type `T` can be placed inside an `Atomic<T>`. After a quick survey of the LLVM and Clang code, architectures can be classified into 3 categories:

- The architecture does not support any form of atomics (mainly microcontroller architectures).
- The architecture supports all atomic operations for integers from i8 to iN (where N is the architecture word/pointer size).
- The architecture supports all atomic operations for integers from i8 to i(N*2).

A new target cfg is added: `target_has_atomic`. It will have multiple values, one for each atomic size supported by the target. For example:

```rust
#[cfg(target_has_atomic = "128")]
static ATOMIC: AtomicU128 = AtomicU128::new(mem::transmute((0u64, 0u64)));
#[cfg(not(target_has_atomic = "128"))]
static ATOMIC: Mutex<(u64, u64)> = Mutex::new((0, 0));

#[cfg(target_has_atomic = "64")]
static COUNTER: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_has_atomic = "64"))]
static COUTNER: AtomicU32 = AtomicU32::new(0);
```

Note that it is not necessary for an architecture to natively support atomic operations for all sizes (`i8`, `i16`, etc) as long as it is able to perform a `compare_exchange` operation with a larger size. All smaller operations can be emulated using that. For example a byte atomic can be emulated by using a `compare_exchange` loop that only modifies a single byte of the value. This is actually how LLVM implements byte-level atomics on MIPS, which only supports word-sized atomics native. Note that the out-of-bounds read is fine here because atomics are aligned and will never cross a page boundary. Since this transformation is performed transparently by LLVM, we do not need to do any extra work to support this.

## Changes to `AtomicPtr`, `AtomicIsize` and `AtomicUsize`

These types will have a `#[cfg(target_has_atomic = "ptr")]` bound added to them. Although these types are stable, this isn't a breaking change because all targets currently supported by Rust will have this type available. This would only affect custom targets, which currently fail to link due to missing compiler-rt symbols anyways.

## Changes to `AtomicBool`

This type will be changes to use an `AtomicU8` internally instead of an `AtomicUsize`, which will allow it to be safely transmuted to a `bool`. This will make it more consistent with the other atomic types that have the same layout as their underlying type. (For example futex code will assume that a `&AtomicI32` can be passed as a `&i32` to the system call)

# Drawbacks
[drawbacks]: #drawbacks

Having certain atomic types get enabled/disable based on the target isn't very nice, but it's unavoidable because support for atomic operations is very architecture-specific.

This approach doesn't directly support for atomic operations on user-defined structs, but this can be emulated using transmutes.

# Alternatives
[alternatives]: #alternatives

One alternative that was discussed in a [previous RFC](https://github.com/rust-lang/rfcs/pull/1505) was to add a generic `Atomic<T>` type. However the consensus was that having unsupported atomic types either fail at monomorphization time or fall back to lock-based implementations was undesirable.

Several other designs have been suggested [here](https://internals.rust-lang.org/t/pre-rfc-extended-atomic-types/3068).

# Unresolved questions
[unresolved]: #unresolved-questions

None

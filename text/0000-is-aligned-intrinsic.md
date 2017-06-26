- Feature Name: is_aligned_intrinsic
- Start Date: 2017-06-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add an intrinsic (`fn is_aligned<T>(*const T) -> bool`) which returns `true` if
a read through the pointer would not fail or suffer from slowdown due to the
alignment of the pointer. There are no further guarantees given about the pointer
it is perfectly valid for `is_aligned(0 as *const T)` to return `true`.

# Motivation
[motivation]: #motivation

The standard library (and most likely many crates) use code like

```rust
let is_aligned = (ptr as usize) & ((1 << (align - 1)) - 1) == 0;
let is_2_byte_aligned = ((ptr as usize + index) & (usize_bytes - 1)) == 0;
let is_t_aligned = ((ptr as usize) % std::mem::align_of::<T>()) == 0;
```

to check whether a pointer is aligned in order to perform optimizations like
reading multiple bytes at once. Not only is this code which is easy to get
wrong, and which is hard to read (and thus increasing the chance of future breakage)
but it also makes it impossible for `miri` to evaluate such statements. This
means that `miri` cannot do utf8-checking, since that code contains such
optimizations. Without utf8-checking, Rustc's future const evaluation would not
be able to convert a `[u8]` into a `str`.

# Detailed design
[design]: #detailed-design

Add a new intrinsic `is_aligned` with the following signature:

```rust
fn is_aligned<T>(*const T) -> bool
```

Calls to `is_aligned` are expanded to

```rust
(ptr as usize) % core::mem::align_of::<T>() == 0
```

on all current platforms.

An Rust backend may decide to expand it differently, but it needs to hold up the
guarantee from `core::mem::align_of`:

> Every valid address of a value of the type `T` must be a multiple of this number.

The intrinsic may be stricter than this and conservatively return `false` for
arbitrary platform specific reasons.

The intrinsic *must* return `true` for pointers to stack or heap allocations
of the type. This means `is_aligned::<T>(Box::<T>::new(t).to_ptr())` is always
true. The same goes for `is_aligned::<T>(&t)`.

The intrinsic is exported in `core::ptr` and `std::ptr` as a safe `is_aligned`
function with the same signature. It is safe to call, since the pointer is
not dereferenced and no operation is applied to it which could overflow.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

On most platforms alignment is a well known concept independent of Rust. The
important thing that has to be taught about this intrinsic is that it may not
ever return "true" no matter which operations are applied to the pointer. The
only guaranteed way to make the intrinsic return to is to give it a non dangling
pointer to an object of the type given to the intrinsic.

This means that `while !is_aligned(ptr) { ... }` should be considered a
potential infinite loop.

A lint could be implemented which detects hand-written alignment checks and
suggests to use the `is_aligned` function instead. 

# Drawbacks
[drawbacks]: #drawbacks

None known to the author.

# Alternatives
[alternatives]: #alternatives

Miri could intercept calls to functions known to do alignment checks on pointers
and roll its own implementation for them. This doesn't scale well and is prone
to errors due to code duplication.

# Unresolved questions
[unresolved]: #unresolved-questions

Should the guarantees be even lower? Calling the intrinsic on the same pointer
twice could suddenly yield `false` even if it yielded `true` before. Since it
should only be used for optimizations this would not be an issue, but be even
more suprising behaviour for no benefit known to the auther.

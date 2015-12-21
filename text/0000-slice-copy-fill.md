- Feature Name: std::slice::copy, slice::fill
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Safe `memcpy` from one slice to another of the same type and length, and a safe
`memset` of a slice of type `T: Copy`.

# Motivation
[motivation]: #motivation

Currently, the only way to quickly copy from one non-`u8` slice to another is to
use a loop, or unsafe methods like `std::ptr::copy_nonoverlapping`. This allows
us to guarantee a `memcpy` for `Copy` types, and is safe. The only way to
`memset` a slice, currently, is a loop, and we should expose a method to allow
people to do this. This also completely gets rid of the point of
`std::slice::bytes`, which means we can remove this deprecated and useless
module.

# Detailed design
[design]: #detailed-design

Add two methods to Primitive Type `slice`.

```rust
impl<T> [T] where T: Copy {
    pub fn fill(&mut self, value: T);
    pub fn copy_from(&mut self, src: &[T]);
}
```

`fill` loops through slice, setting each member to value. This will lower to a
memset in all possible cases. It is defined behavior to call `fill` on a slice
which has uninitialized members, and `self` is guaranteed to be fully filled
afterwards.

`copy` panics if `src.len() != self.len()`, then `memcpy`s the members into 
`self` from `src`. Calling `copy_from` is semantically equivalent to a `memcpy`;
`self` can have uninitialized members, and `self` is guaranteed to be fully filled
afterwards. This means, for example, that the following is fully defined:

```rust
let s1: [u8; 16] = unsafe { std::mem::uninitialized() };
let s2: [u8; 16] = unsafe { std::mem::uninitialized() };
s1.fill(42);
s2.copy_from(&s1);
println!("{}", s2);
```

And the program will print 16 '8's.

# Drawbacks
[drawbacks]: #drawbacks

Two new methods on `slice`. `[T]::fill` *will not* be lowered to a `memset` in
any case where the bytes of `value` are not all the same, as in

```rust
// let points: [f32; 16];
points.fill(1.0); // This is not lowered to a memset (However, it is lowered to
                       // a simd loop, which is what a memset is, in reality)
```

Also, `copy_from` has it's arguments in a different order from it's most similar
`unsafe` alternative, `std::ptr::copy_nonoverlapping`. This is due to an
unfortunate error that cannot be solved with the now stable
`copy_nonoverlapping`, and the design decision should not be extended to
`copy_from`.

# Alternatives
[alternatives]: #alternatives

We could name these functions something else. `fill`, for example, could be
called `set`, `fill_from`, or `fill_with`.

`copy_from` could be called `copy_to`, and have the order of the arguments
switched around. This is a bad idea, as `copy_from` has a natural connection to
`dst = src` syntax.

`memcpy` is also pretty weird, here. Panicking if the lengths differ is
different from what came before; I believe it to be the safest path, because I
think I'd want to know, personally, if I'm passing the wrong lengths to copy.
However, `std::slice::bytes::copy_memory`, the function I'm basing this on, only
panics if `dst.len() < src.len()`. So... room for discussion, here.

`fill` and `copy_from` could both be free functions, and were in the
original draft of this document. However, overwhelming support for these as
methods has meant that these have become methods.

These are necessary, in my opinion. Much unsafe code has been written because
these do not exist.

# Unresolved questions
[unresolved]: #unresolved-questions

None, as far as I can tell.

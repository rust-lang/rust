- Feature Name: std::slice::{ copy, set };
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

Add two functions to `std::slice`.

```rust
pub fn set<T: Copy>(slice: &mut [T], value: T);
pub fn copy<T: Copy>(src: &[T], dst: &mut [T]);
```

`set` loops through slice, setting each member to value. This will lower to a
memset in all possible cases.

`copy` panics if `src.len() != dst.len()`, then `memcpy`s the members from
`src` to `dst`.

# Drawbacks
[drawbacks]: #drawbacks

Two new functions in `std::slice`. `std::slice::set` *will not* be lowered to a
`memset` in any case where the bytes of `value` are not all the same, as in

```rust
// let points: [f32; 16];
std::slice::set(&mut points, 1.0); // This is not lowered to a memset
                                   // (However, it is lowered to a simd loop,
                                   //  which is what a memset is, in reality)
```

# Alternatives
[alternatives]: #alternatives

We could name these functions something different.

`memcpy` is also pretty weird, here. Panicking if the lengths differ is
different from what came before; I believe it to be the safest path, because I
think I'd want to know, personally, if I'm passing the wrong lengths to copy.
However, `std::slice::bytes::copy_memory`, the function I'm basing this on, only
panics if `dst.len() < src.len()`. So... room for discussion, here.

These are necessary functions, in the opinion of the author.

# Unresolved questions
[unresolved]: #unresolved-questions

None, as far as I can tell.

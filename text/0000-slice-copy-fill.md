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

Add one function to `std::slice`.

```rust
pub fn copy<T: Copy>(src: &[T], dst: &mut [T]);
```

and one function to Primitive Type `slice`.

```rust
impl<T> [T] where T: Copy {
    pub fn fill(&mut self, value: T);
}
```

`fill` loops through slice, setting each member to value. This will lower to a
memset in all possible cases. It is defined to call `fill` on a slice which has
uninitialized members.

`copy` panics if `src.len() != dst.len()`, then `memcpy`s the members from
`src` to `dst`.

# Drawbacks
[drawbacks]: #drawbacks

One new function in `std::slice`, and one new method on `slice`. `[T]::fill`
*will not* be lowered to a `memset` in any case where the bytes of `value` are
not all the same, as in

```rust
// let points: [f32; 16];
points.fill(1.0); // This is not lowered to a memset (However, it is lowered to
                  // a simd loop, which is what a memset is, in reality)
```

# Alternatives
[alternatives]: #alternatives

We could name these functions something else. `fill`, for example, could be
called `set`.

`memcpy` is also pretty weird, here. Panicking if the lengths differ is
different from what came before; I believe it to be the safest path, because I
think I'd want to know, personally, if I'm passing the wrong lengths to copy.
However, `std::slice::bytes::copy_memory`, the function I'm basing this on, only
panics if `dst.len() < src.len()`. So... room for discussion, here.

`fill` could be a free function, and `copy` could be a method. It is the
opinion of the author that `copy` is best as a free function, as it is
non-obvious which should be the "owner", `dst` or `src`. `fill` is more
obviously a method.

These are necessary functions, in the opinion of the author.

# Unresolved questions
[unresolved]: #unresolved-questions

None, as far as I can tell.

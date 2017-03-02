- Feature Name: slice\_copy\_from
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: [rust-lang/rfcs#1419](https://github.com/rust-lang/rfcs/pull/1419)
- Rust Issue: [rust-lang/rust#31755](https://github.com/rust-lang/rust/issues/31755)

# Summary
[summary]: #summary

Safe `memcpy` from one slice to another of the same type and length.

# Motivation
[motivation]: #motivation

Currently, the only way to quickly copy from one non-`u8` slice to another is to
use a loop, or unsafe methods like `std::ptr::copy_nonoverlapping`. This allows
us to guarantee a `memcpy` for `Copy` types, and is safe.

# Detailed design
[design]: #detailed-design

Add one method to Primitive Type `slice`.

```rust
impl<T> [T] where T: Copy {
    pub fn copy_from_slice(&mut self, src: &[T]);
}
```

`copy_from_slice` asserts that `src.len() == self.len()`, then `memcpy`s the
members into `self` from `src`. Calling `copy_from_slice` is semantically
equivalent to a `memcpy`.  `self` shall have exactly the same members as `src`
after a call to `copy_from_slice`.

# Drawbacks
[drawbacks]: #drawbacks

One new method on `slice`.

# Alternatives
[alternatives]: #alternatives

`copy_from_slice` could be called `copy_to`, and have the order of the arguments
switched around. This would follow `ptr::copy_nonoverlapping` ordering, and not
`dst = src` or `.clone_from_slice()` ordering.

`copy_from_slice` could panic only if `dst.len() < src.len()`. This would be the
same as what came before, but we would also lose the guarantee that an
uninitialized slice would be fully initialized.

`copy_from_slice` could be a free function, as it was in the original draft of
this document. However, there was overwhelming support for it as a method.

`copy_from_slice` could be not merged, and `clone_from_slice` could be
specialized to `memcpy` in cases of `T: Copy`. I think it's good to have a
specific function to do this, however, which asserts that `T: Copy`.

# Unresolved questions
[unresolved]: #unresolved-questions

None, as far as I can tell.

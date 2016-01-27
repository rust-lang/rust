- Feature Name: slice\_copy\_from
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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
    pub fn copy_from(&mut self, src: &[T]);
}
```

`copy_from` asserts that `src.len() == self.len()`, then `memcpy`s the members into 
`self` from `src`. Calling `copy_from` is semantically equivalent to a `memcpy`.
`self` shall have exactly the same members as `src` after a call to `copy_from`.

# Drawbacks
[drawbacks]: #drawbacks

One new method on `slice`.

# Alternatives
[alternatives]: #alternatives

`copy_from` could be known as `copy_from_slice`, which would follow
`clone_from_slice`.

`copy_from` could be called `copy_to`, and have the order of the arguments
switched around. This would follow `ptr::copy_nonoverlapping` ordering, and not
`dst = src` or `.clone_from()` ordering.

`copy_from` could panic only if `dst.len() < src.len()`. This would be the same
as what came before, but we would also lose the guarantee that an uninitialized
slice would be fully initialized.

`copy_from` could be a free function, as it was in the original draft of this
document. However, there was overwhelming support for it as a method.

# Unresolved questions
[unresolved]: #unresolved-questions

None, as far as I can tell.

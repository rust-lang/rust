- Start Date: 2014-06-09
- RFC PR: [rust-lang/rfcs#112](https://github.com/rust-lang/rfcs/pull/112)
- Rust Issue: [rust-lang/rust#10504](https://github.com/rust-lang/rust/issues/10504)

# Summary

Remove the coercion from `Box<T>` to `&mut T` from the language.

# Motivation

Currently, the coercion between `Box<T>` to `&mut T` can be a hazard because it can lead to surprising mutation where it was not expected.

# Detailed design

The coercion between `Box<T>` and `&mut T` should be removed.

Note that methods that take `&mut self` can still be called on values of type `Box<T>` without any special referencing or dereferencing. That is because the semantics of auto-deref and auto-ref conspire to make it work: the types unify after one autoderef followed by one autoref.

# Drawbacks

Borrowing from `Box<T>` to `&mut T` may be convenient.

# Alternatives

An alternative is to remove `&T` coercions as well, but this was decided against as they are convenient.

The impact of not doing this is that the coercion will remain.

# Unresolved questions

None.

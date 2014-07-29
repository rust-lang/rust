- Start Date: 2014-06-25
- RFC PR: [rust-lang/rfcs#139](https://github.com/rust-lang/rfcs/pull/139)
- Rust Issue: [rust-lang/rust#10504](https://github.com/rust-lang/rust/issues/10504)

# Summary

Remove the coercion from `Box<T>` to `&T` from the language.

# Motivation

The coercion between `Box<T>` to `&T` is not replicable by user-defined smart pointers and has been found to be rarely used [1]. We already removed the coercion between `Box<T>` and `&mut T` in RFC 33.

# Detailed design

The coercion between `Box<T>` and `&T` should be removed.

Note that methods that take `&self` can still be called on values of type `Box<T>` without any special referencing or dereferencing. That is because the semantics of auto-deref and auto-ref conspire to make it work: the types unify after one autoderef followed by one autoref.

# Drawbacks

Borrowing from `Box<T>` to `&T` may be convenient.

# Alternatives

The impact of not doing this is that the coercion will remain.

# Unresolved questions

None.

[1]: https://github.com/rust-lang/rust/pull/15171

- Feature Name: N/A
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

With specialization on the way, we need to talk about the semantics of
`<T as Clone>::clone() where T: Copy`.

It's generally been an unspoken rule of Rust that a `clone` of a `Copy` type is
equivalent to a `memcpy` of that type; however, that fact is not documented
anywhere. This fact should be in the documentation for the `Clone` trait, just
like the fact that `T: Eq` should implement `a == b == c == a` rules.

# Motivation
[motivation]: #motivation

Currently, `Vec::clone()` is implemented by creating a new `Vec`, and then
cloning all of the elements from one into the other. This is slow in debug mode,
and may not always be optimized (although it often will be). Specialization
would allow us to simply `memcpy` the values from the old `Vec` to the new
`Vec`. However, if we don't actually specify this, we will not be able to do
this.

# Detailed design
[design]: #detailed-design

Simply add something like the following sentence to the documentation for the
`Clone` trait:

"If `T: Copy`, `x: T`, and `y: &T`, then `let x = y.clone()` is equivalent to
`let x = *y`;"

# Drawbacks
[drawbacks]: #drawbacks

This is a breaking change, technically, although it breaks code that was
malformed in the first place.

# Alternatives
[alternatives]: #alternatives

The alternative is that, for each type and function we would like to specialize
in this way, we document this separately. This is how we started off with
`clone_from_slice`.

# Unresolved questions
[unresolved]: #unresolved-questions

What the exact sentence should be.

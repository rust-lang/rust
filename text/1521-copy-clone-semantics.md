- Feature Name: N/A
- Start Date: 01 March, 2016
- RFC PR: [rust-lang/rfcs#1521](https://github.com/rust-lang/rfcs/pull/1521)
- Rust Issue: [rust-lang/rust#33416](https://github.com/rust-lang/rust/issues/33416)

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
`Vec` in the case of `T: Copy`. However, if we don't specify this, we will not
be able to, and we will be stuck looping over every value.

It's always been the intention that `Clone::clone == ptr::read for T: Copy`; see
[issue #23790][issue-copy]: "It really makes sense for `Clone` to be a
supertrait of `Copy` -- `Copy` is a refinement of `Clone` where `memcpy`
suffices, basically." This idea was also implicit in accepting
[rfc #0839][rfc-extend] where "[B]ecause Copy: Clone, it would be backwards
compatible to upgrade to Clone in the future if demand is high enough."

# Detailed design
[design]: #detailed-design

Specify that `<T as Clone>::clone(t)` shall be equivalent to `ptr::read(t)`
where `T: Copy, t: &T`. An implementation that does not uphold this *shall not*
result in undefined behavior; `Clone` is not an `unsafe trait`.

Also add something like the following sentence to the documentation for the
`Clone` trait:

"If `T: Copy`, `x: T`, and `y: &T`, then `let x = y.clone();` is equivalent to
`let x = *y;`. Manual implementations must be careful to uphold this."

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

What the exact wording should be.

[issue-copy]: https://github.com/rust-lang/rust/issues/23790
[rfc-extend]: https://github.com/rust-lang/rfcs/blob/master/text/0839-embrace-extend-extinguish.md

- Feature Name: inclusive_range_syntax
- Start Date: 2015-07-07
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Allow a `x...y` expression to create an inclusive range.

# Motivation

There are several use-cases for inclusive ranges, that semantically
include both end-points. For example, iterating from `0_u8` up to and
including some number `n` can be done via `for _ in 0..n + 1` at the
moment, but this will fail if `n` is `255`. Furthermore, some iterable
things only have a successor operation that is sometimes sensible,
e.g., `'a'..'{'` is equivalent to the inclusive range `'a'...'z'`:
there's absolutely no reason that `{` is after `z` other than a quirk
of the representation.

The `...` syntax mirrors the current `..` used for exclusive ranges:
more dots means more elements.

# Detailed design

`std::ops` defines

```rust
pub struct RangeInclusive<T> {
    pub start: T,
    pub end: T,
}
```

Writing `a...b` in an expression desugars to `std::ops::RangeInclusive
{ start: a, end: b }`.

This struct implements the standard traits (`Clone`, `Debug` etc.),
but, unlike the other `Range*` types, does not implement `Iterator`
directly, since it cannot do so correctly without more internal
state. It can implement `IntoIterator` that converts it into an
iterator type that contains the necessary state.

The use of `...` in a pattern remains as testing for inclusion
within that range, *not* a struct match.

The author cannot forsee problems with breaking backward
compatibility. In particular, one tokenisation of syntax like `1...`
now would be `1. ..` i.e. a floating point number on the left, however, fortunately,
it is actually tokenised like `1 ...`, and is hence an error.

# Drawbacks

There's a mismatch between pattern-`...` and expression-`...`, in that
the former doesn't undergo the same desugaring as the
latter. (Although they represent essentially the same thing
semantically.)

The `...` vs. `..` distinction is the exact inversion of Ruby's syntax.

Only implementing `IntoIterator` means uses of it in iterator chains
look like `(a...b).into_iter().collect()` instead of
`(a..b).collect()` as with exclusive ones (although this doesn't
affect `for` loops: `for _ in a...b` works fine).

# Alternatives

An alternate syntax could be used, like
`..=`. [There has been discussion][discuss], but there wasn't a clear
winner.

[discuss]: https://internals.rust-lang.org/t/vs-for-inclusive-ranges/1539

This RFC doesn't propose non-double-ended syntax, like `a...`, `...b`
or `...` since it isn't clear that this is so useful. Maybe it is.

# Unresolved questions

None so far.

- Feature Name: inclusive_range_syntax
- Start Date: 2015-07-07
- RFC PR: [rust-lang/rfcs#1192](https://github.com/rust-lang/rfcs/pull/1192)
- Rust Issue: [rust-lang/rust#28237](https://github.com/rust-lang/rust/issues/28237)

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
pub enum RangeInclusive<T> {
  Empty {
    at: T,
  },
  NonEmpty {
    start: T,
    end: T,
  }
}

pub struct RangeToInclusive<T> {
    pub end: T,
}
```

Writing `a...b` in an expression desugars to `std::ops::RangeInclusive::NonEmpty { start: a, end: b }`. Writing `...b` in an
expression desugars to `std::ops::RangeToInclusive { end: b }`.

`RangeInclusive` implements the standard traits (`Clone`, `Debug`
etc.), and implements `Iterator`. The `Empty` variant is to allow the
`Iterator` implementation to work without hacks (see Alternatives).

The use of `...` in a pattern remains as testing for inclusion
within that range, *not* a struct match.

The author cannot forsee problems with breaking backward
compatibility. In particular, one tokenisation of syntax like `1...`
now would be `1. ..` i.e. a floating point number on the left,
however, fortunately, it is actually tokenised like `1 ...`, and is
hence an error with the current compiler.

# Drawbacks

There's a mismatch between pattern-`...` and expression-`...`, in that
the former doesn't undergo the same desugaring as the
latter. (Although they represent essentially the same thing
semantically.)

The `...` vs. `..` distinction is the exact inversion of Ruby's syntax.

Having an extra field in a language-level desugaring, catering to one
library use-case is a little non-"hygienic". It is especially strange
that the field isn't consistent across the different `...`
desugarings.

# Alternatives

An alternate syntax could be used, like
`..=`. [There has been discussion][discuss], but there wasn't a clear
winner.

[discuss]: https://internals.rust-lang.org/t/vs-for-inclusive-ranges/1539

This RFC doesn't propose non-double-ended syntax, like `a...`, `...b`
or `...` since it isn't clear that this is so useful. Maybe it is.

The `Empty` variant could be omitted, leaving two options:

- `RangeInclusive` could be a struct including a `finished` field.
- `a...b` only implements `IntoIterator`, not `Iterator`, by
  converting to a different type that does have the field. However,
  this means that `a.. .b` behaves differently to `a..b`, so
  `(a...b).map(|x| ...)` doesn't work (the `..` version of that is
  used reasonably often, in the author's experience)
- `a...b` can implement `Iterator` for types that can be stepped
  backwards: the only case that is problematic things cases like
  `x...255u8` where the endpoint is the last value in the type's
  range. A naive implementation that just steps `x` and compares
  against the second value will never terminate: it will yield 254
  (final state: `255...255`), 255 (final state: `0...255`), 0 (final
  state: `1...255`). I.e. it will wrap around because it has no way to
  detect whether 255 has been yielded or not. However, implementations
  of `Iterator` can detect cases like that, and, after yielding `255`,
  backwards-step the second piece of state to `255...254`.

  This means that `a...b` can only implement `Iterator` for types that
  can be stepped backwards, which isn't always guaranteed, e.g. types
  might not have a unique predecessor (walking along a DAG).

# Unresolved questions

None so far.

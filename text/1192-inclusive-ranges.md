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
pub struct RangeInclusive<T> {
    pub start: T,
    pub end: T,
}

pub struct RangeToInclusive<T> {
    pub end: T,
}
```

Writing `a...b` in an expression desugars to
`std::ops::RangeInclusive { start: a, end: b }`. Writing `...b` in an
expression desugars to `std::ops::RangeToInclusive { end: b }`.

`RangeInclusive` implements the standard traits (`Clone`, `Debug`
etc.), and implements `Iterator`.

The use of `...` in a pattern remains as testing for inclusion
within that range, *not* a struct match.

The author cannot forsee problems with breaking backward
compatibility. In particular, one tokenisation of syntax like `1...`
now would be `1. ..` i.e. a floating point number on the left,
however, fortunately, it is actually tokenised like `1 ...`, and is
hence an error with the current compiler.

This `struct` definition is maximally consistent with the existing `Range`.
`a..b` and `a...b` are the same size and have the same fields, just with
the expected difference in semantics.

The range `a...b` contains all `x` where `a <= x && x <= b`.  As such, an
inclusive range is non-empty _iff_ `a <= b`.  When the range is iterable,
a non-empty range will produce at least one item when iterated.  Because
`T::MAX...T::MAX` is a non-empty range, the iteration needs extra handling
compared to a half-open `Range`.  As such, `.next()` on an empty range
`y...y` will produce the value `y` and adjust the range such that
`!(start <= end)`.  Providing such a range is not a burden on the `T` type as
any such range is acceptable, and only `PartialOrd` is required so
it can be satisfied with an incomparable value `n` with `!(n <= n)`.
A caller must not, in general, expect any particular `start` or `end`
after iterating, and is encouraged to detect empty ranges with
`ExactSizeIterator::is_empty` instead of by observing fields directly.

Note that because ranges are not required to be well-formed, they have a
much stronger bound than just needing successor function: they require a
`b is-reachable-from a` predicate (as `a <= b`). Providing that efficiently
for a DAG walk, or even a simpler forward list walk, is a substantially
harder thing to do than providing a pair `(x, y)` such that `!(x <= y)`.

Implementation note: For currently-iterable types, the initial implementation
of this will have the range become `1...0` after yielding the final value,
as that can be done using the `replace_one` and `replace_zero` methods on
the existing (but unstable) [`Step` trait][step_trait].  It's expected,
however, that the trait will change to allow more type-appropriate `impl`s.
For example, a `num::BigInt` may rather become empty by incrementing `start`,
as `Range` does, since it doesn't to need to worry about overflow.  Even for
primitives, it could be advantageous to choose a different implementation,
perhaps using `.overflowing_add(1)` and swapping on overflow, or `a...a`
could become `(a+1)...a` where possible and `a...(a-1)` otherwise.

[step_trait]: https://github.com/rust-lang/rust/issues/27741

# Drawbacks

There's a mismatch between pattern-`...` and expression-`...`, in that
the former doesn't undergo the same desugaring as the
latter. (Although they represent essentially the same thing
semantically.)

The `...` vs. `..` distinction is the exact inversion of Ruby's syntax.

This proposal makes the post-iteration values of the `start` and `end` fields
constant, and thus useless.  Some of the alternatives would expose the
last value returned from the iteration, through a more complex interface.

# Alternatives

An alternate syntax could be used, like
`..=`. [There has been discussion][discuss], but there wasn't a clear
winner.

[discuss]: https://internals.rust-lang.org/t/vs-for-inclusive-ranges/1539

This RFC proposes single-ended syntax with only an end, `...b`, but not
with only a start (`a...`) or unconstrained `...`. This balance could be
reevaluated for usefulness and conflicts with other proposed syntax.

- `RangeInclusive` could be a struct including a `finished` field.
  This makes it easier for the struct to always be iterable, as the extra
  field is set once the ends match.  But having the extra field in a
  language-level desugaring, catering to one library use-case is a little
  non-"hygienic". It is especially strange that the field isn't consistent
  across the different `...` desugarings.  And the presence of the public
  field encourages checkinging it, which can be misleading as
  `r.finished == false` does not guarantee that `r.count() > 0`.
- `RangeInclusive` could be an enum with `Empty` and `NonEmpty` variants.
  This is cleaner than the `finished` field, but still has the problem that
  there's no invariant maintained: while an `Empty` range is definitely empty,
  a `NonEmpty` range might actually be empty.  And requiring matching on every
  use of the type is less ergonomic.  For example, the clamp RFC would
  naturally use a `RangeInclusive` parameter, but because it still needs
  to `assert!(start <= end)` in the `NonEmpty` arm, the noise of the `Empty`
  vs `NonEmpty` match provides it no value.
- `a...b` only implements `IntoIterator`, not `Iterator`, by
  converting to a different type that does have the field. However,
  this means that `a.. .b` behaves differently to `a..b`, so
  `(a...b).map(|x| ...)` doesn't work (the `..` version of that is
  used reasonably often, in the author's experience)
- The name of the `end` field could be different, perhaps `last`, to reflect
  its different (inclusive) semantics from the `end` (exclusive) field on
  the other ranges.

# Unresolved questions

None so far.

# Amendments

* In rust-lang/rfcs#1320, this RFC was amended to change the `RangeInclusive`
  type from a struct with a `finished` field to an enum.
* In rust-lang/rfcs#1980, this RFC was amended to change the `RangeInclusive`
  type from an enum to a struct with just `start` and `end` fields.

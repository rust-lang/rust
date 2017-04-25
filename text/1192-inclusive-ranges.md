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
etc.), and implements `Iterator`. The `Empty` variant is to allow the
`Iterator` implementation to work without hacks (see Alternatives).

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

# Drawbacks

There's a mismatch between pattern-`...` and expression-`...`, in that
the former doesn't undergo the same desugaring as the
latter. (Although they represent essentially the same thing
semantically.)

The `...` vs. `..` distinction is the exact inversion of Ruby's syntax.

Not having a separate marker for `finished` or `empty` implies a requirement
on `T` that it's possible to provide values such that `b...a` is an empty
range.  But a separate marker is a false invariant: whether a `finished`
field on the struct or a `Empty` variant of an enum, the range `10...0` still
desugars to a `RangeInclusive` with `finised: false` or of the `NonEmpty`
variant.  And the fields are public, so even fixing the desugar cannot
guarantee the invariant.  As a result, all code using a `RangeInclusive`
must still check whether a "`NonEmpty`" or "un`finished`" is actually finished.
The "can produce an empty range" requirement is not a hardship.  It's trivial
for anything that can be stepped forward and backward, as all things which are
iterable in `std` are today.  But ther are other possibilities as well.  The
proof-of-concept implementation for this change is done using the `replace_one`
and `replace_zero` methods of the (existing but unstable) `Step` trait, as
`1...0` is of course an empty range.  Something weirder, like walking along a
DAG, could use the fact that `PartialOrd` is sufficient, and produce a range
similar in character to `NaN...NaN`, which is empty as `(NaN <= NaN) == false`.
The exact details of what is required to make a range iterable is outside the
scope of this RFC, and will be decided in the [`step_by` issue][step_by].

Note that iterable ranges today have a much stronger bound than just
steppability: they require a `b is-reachable-from a` predicate (as `a <= b`).
Providing that efficiently for a DAG walk, or even a simpler forward list
walk, is a substantially harder thing to do that providing a pair `(x, y)`
such that `!(x <= y)`.

[step_by]: https://github.com/rust-lang/rust/issues/27741

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
  across the different `...` desugarings.
- `RangeInclusive` could be an enum with `Empty` and `NonEmpty` variants.
  This is cleaner than the `finished` field, but makes all uses of the
  type substantially more complex.  For example, the clamp RFC would
  naturally use a `RangeInclusive` parameter, but then the
  unreliable-`Empty` vs `NonEmpty` distinction provides no value.  It does
  prevent looking at `start` after iteration has completed, but that is
  of questionable value when `Range` allows it without issue, and disallowing
  looking at `start` while allowing looking at `end` feels inconsistent.
- `a...b` only implements `IntoIterator`, not `Iterator`, by
  converting to a different type that does have the field. However,
  this means that `a.. .b` behaves differently to `a..b`, so
  `(a...b).map(|x| ...)` doesn't work (the `..` version of that is
  used reasonably often, in the author's experience)

# Unresolved questions

None so far.

# Amendments

* In rust-lang/rfcs#1320, this RFC was amended to change the `RangeInclusive`
  type from a struct with a `finished` field to an enum.
* In rust-lang/rfcs#1980, this RFC was amended to change the `RangeInclusive`
  type from an enum to a struct with just `start` and `end` fields.

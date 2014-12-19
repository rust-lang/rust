- Start Date: 2014-12-13
- RFC PR: [520](https://github.com/rust-lang/rfcs/pull/520)
- Rust Issue: [19999](https://github.com/rust-lang/rust/issues/19999)

# Summary

Under this RFC, the syntax to specify the type of a fixed-length array
containing `N` elements of type `T` would be changed to `[T; N]`. Similarly, the
syntax to construct an array containing `N` duplicated elements of value `x`
would be changed to `[x; N]`.

# Motivation

[RFC 439](https://github.com/rust-lang/rfcs/blob/master/text/0439-cmp-ops-reform.md)
(cmp/ops reform) has resulted in an ambiguity that must be resolved. Previously,
an expression with the form `[x, ..N]` would unambiguously refer to an array
containing `N` identical elements, since there would be no other meaning that
could be assigned to `..N`. However, under RFC 439, `..N` should now desugar to
an object of type `RangeTo<T>`, with `T` being the type of `N`.

In order to resolve this ambiguity, there must be a change to either the syntax
for creating an array of repeated values, or the new range syntax. This RFC
proposes the former, in order to preserve existing functionality while avoiding
modifications that would make the range syntax less intuitive.

# Detailed design

The syntax `[T, ..N]` for specifying array types will be replaced by the new
syntax `[T; N]`.

In the expression `[x, ..N]`, the `..N` will refer to an expression of type
`RangeTo<T>` (where `T` is the type of `N`). As with any other array of two
elements, `x` will have to be of the same type, and the array expression will be
of type `[RangeTo<T>; 2]`.

The expression `[x; N]` will be equivalent to the old meaning of the syntax
`[x, ..N]`. Specifically, it will create an array of length `N`, each element of
which has the value `x`.

The effect will be to convert uses of arrays such as this:

```rust
let a: [uint, ..2] = [0u, ..2];
```

to this:

```rust
let a: [uint; 2] = [0u; 2];
```

## Match patterns

In match patterns, `..` is always interpreted as a wildcard for constructor
arguments (or for slice patterns under the `advanced_slice_patterns` feature
gate). This RFC does not change that. In a match pattern, `..` will always be
interpreted as a wildcard, and never as sugar for a range constructor.

## Suggested implementation

While not required by this RFC, one suggested transition plan is as follows:

- Implement the new syntax for `[T; N]`/`[x; N]` proposed above.

- Issue deprecation warnings for code that uses `[T, ..N]`/`[x, ..N]`, allowing
  easier identification of code that needs to be transitioned.

- When RFC 439 range literals are implemented, remove the deprecated syntax and
  thus complete the implementation of this RFC.

# Drawbacks

## Backwards incompatibility

- Changing the method for specifying an array size will impact a large amount of
  existing code. Code conversion can probably be readily automated, but will
  still require some labor.

## Implementation time

This proposal is submitted very close to the anticipated release of Rust
1.0. Changing the array repeat syntax is likely to require more work than
changing the range syntax specified in RFC 439, because the latter has not yet
been implemented.

However, this decision cannot be reasonably postponed. Many users have expressed
a preference for implementing the RFC 439 slicing syntax as currently specified
rather than preserving the existing array repeat syntax. This cannot be resolved
in a backwards-compatible manner if the array repeat syntax is kept.

# Alternatives

Inaction is not an alternative due to the ambiguity introduced by RFC 439. Some
resolution must be chosen in order for the affected modules in `std` to be
stabilized.

## Retain the type syntax only

In theory, it seems that the type syntax `[T, ..N]` could be retained, while
getting rid of the expression syntax `[x, ..N]`. The problem with this is that,
if this syntax was removed, there is currently no way to define a macro to
replace it.

Retaining the current type syntax, but changing the expression syntax, would
make the language somewhat more complex and inconsistent overall. There seem to
be no advocates of this alternative so far.

## Different array repeat syntax

The comments in [pull request #498](https://github.com/rust-lang/rfcs/pull/498)
mentioned many candidates for new syntax other than the `[x; N]` form in this
RFC. The comments on the pull request of this RFC mentioned many more.

- Instead of using `[x; N]`, use `[x for N]`.

    - This use of `for` would not be exactly analogous to existing `for` loops,
      because those accept an iterator rather than an integer. To a new user,
      the expression `[x for N]` would resemble a list comprehension
      (e.g. Python's syntax is `[expr for i in iter]`), but in fact it does
      something much simpler.
    - It may be better to avoid uses of `for` that could complicate future
      language features, e.g. returning a value other than `()` from loops, or
      some other syntactic sugar related to iterators. However, the risk of
      actual ambiguity is not that high.

- Introduce a different symbol to specify array sizes, e.g. `[T # N]`,
  `[T @ N]`, and so forth.

- Introduce a keyword rather than a symbol. There are many other options, e.g.
  `[x by N]`. The original version of this proposal was for `[N of x]`, but this
  was deemed to complicate parsing too much, since the parser would not know
  whether to expect a type or an expression after the opening bracket.

- Any of several more radical changes.

## Change the range syntax

The main problem here is that there are no proposed candidates that seem as
clear and ergonomic as `i..j`. The most common alternative for slicing in other
languages is `i:j`, but in Rust this simply causes an ambiguity with a different
feature, namely type ascription.

## Limit range syntax to the interior of an index (use `i..j` for slicing only)

This resolves the issue since indices can be distinguished from arrays. However,
it removes some of the benefits of RFC 439. For instance, it removes the
possibility of using `for i in 1..10` to loop.

## Remove `RangeTo` from RFC 439

The proposal in pull request #498 is to remove the sugar for `RangeTo` (i.e.,
`..j`) while retaining other features of RFC 439. This is the simplest
resolution, but removes some convenience from the language. It is also
counterintuitive, because `RangeFrom` (i.e. `i..`) is retained, and because `..`
still has several different meanings in the language (ranges, repetition, and
pattern wildcards).

# Unresolved questions

## Match patterns

There will still be two semantically distinct uses of `..`, for the RFC 439
range syntax and for wildcards in patterns. This could be considered harmful
enough to introduce further changes to separate the two. Or this could be
considered innocuous enough to introduce some additional range-related meaning
for `..` in certain patterns.

It is possible that the new syntax `[x; N]` could itself be used within
patterns.

This RFC does not attempt to address any of these issues, because the current
pattern syntax does not allow use of the repeated array syntax, and does not
contain an ambiguity.

## Behavior of `for` in array expressions

It may be useful to allow `for` to take on a new meaning in array expressions.
This RFC keeps this possibility open, but does not otherwise propose any
concrete changes to move towards or away from this feature.

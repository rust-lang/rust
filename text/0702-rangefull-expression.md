- Start Date: 2015-01-21
- RFC PR: [#702](https://github.com/rust-lang/rfcs/pull/702)
- Rust Issue: [#21879](https://github.com/rust-lang/rust/issues/21879)

# Summary

Add the syntax `..` for `std::ops::RangeFull`.

# Motivation

Range expressions `a..b`, `a..` and `..b` all have dedicated syntax and
produce first-class values. This means that they will be usable and
useful in custom APIs, so for consistency, the fourth slicing range,
`RangeFull`, could have its own syntax `..`

# Detailed design

`..` will produce a `std::ops::RangeFull` value when it is used in an
expression. This means that slicing the whole range of a sliceable
container is written `&foo[..]`.

We should remove the old `&foo[]` syntax for consistency. Because of
this breaking change, it would be best to change this before Rust 1.0.

As previously stated, when we have range expressions in the language,
they become convenient to use when stating ranges in an API.

@Gankro fielded ideas where
methods like for example `.remove(index) -> element` on a collection
could be generalized by accepting either indices or ranges. Today's `.drain()`
could be expressed as `.remove(..)`.

Matrix or multidimensional array APIs can use the range expressions for
indexing and/or generalized slicing and `..` represents selecting a full axis
in a multidimensional slice, i.e. `(1..3, ..)` slices the first axis and
preserves the second.

Because of deref coercions, the very common conversions of String or Vec to
slices don't need to use slicing syntax at all, so the change in verbosity from
`[]` to `[..]` is not a concern.

# Drawbacks

* Removing the slicing syntax `&foo[]` is a breaking change.

* `..` already appears in patterns, as in this example: 
  `if let Some(..) = foo { }`. This is not a conflict per se, but the
  same syntax element is used in two different ways in Rust.

# Alternatives

* We could add this syntax later, but we would end up with duplicate
  slicing functionality using `&foo[]` and `&foo[..]`.

* `0..` could replace `..` in many use cases (but not for ranges in
  ordered maps).

# Unresolved questions

Any parsing questions should already be mostly solved because of the
`a..` and `..b` cases.

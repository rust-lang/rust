- Feature Name: `slice_string_symmetry`
- Start Date: 2015-06-06
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add some methods that already exist on slices to strings and vice versa.
Specifically, the following methods should be added:

- `str::chunks`
- `str::windows`
- `str::into_string`
- `String::into_boxed_slice`
- `<[T]>::subslice_offset`

# Motivation

Conceptually, strings and slices are similar types. Many methods are already
shared between the two types due to their similarity. However, not all methods
are shared between the types, even though many could be. This is a little
unexpected and inconsistent. Because of that, this RFC proposes to remedy this
by adding a few methods to both strings and slices to even out these two types’
available methods.

# Detailed design

Add the following methods to `str`, presumably as inherent methods:

- `chunks(&self, n: usize) -> Chunks`: Returns an iterator that yields the
  *characters* (not bytes) of the string in groups of `n` at a time. Iterator
  element type: `&str`.

- `windows(&self, n: usize) -> Windows`: Returns an iterator over all contiguous
  windows of character length `n`. Iterator element type: `&str`.

- `into_string(self: Box<str>) -> String`: Returns `self` as a `String`. This is
  equivalent to `[T]`’s `into_vec`.

`split_at(&self, mid: usize) -> (&str, &str)` would also be on this list, but
there is [an existing RFC](https://github.com/rust-lang/rfcs/pull/1123) for it.

Add the following method to `String` as an inherent method:

- `into_boxed_slice(self) -> Box<str>`: Returns `self` as a `Box<str>`,
  reallocating to cut off any excess capacity if needed. This is required to
  provide a safe means of creating `Box<str>`.

Add the following method to `[T]` (for all `T`), presumably as an inherent
method:

- `subslice_offset(&self, inner: &[T]) -> usize`: Returns the offset (in
  elements) of an inner slice relative to an outer slice. Panics of `inner` is
  not contained within `self`.

# Drawbacks

- `str::subslice_offset` is already unstable, so creating a similar method on
  `[T]` is perhaps not such a good idea.

# Alternatives

- Do a subset of the proposal. For example, the `Box<str>`-related methods could
  be removed.

# Unresolved questions

None.

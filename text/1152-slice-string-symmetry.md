- Feature Name: `slice_string_symmetry`
- Start Date: 2015-06-06
- RFC PR: [rust-lang/rfcs#1152](https://github.com/rust-lang/rfcs/pull/1152)
- Rust Issue: [rust-lang/rust#26697](https://github.com/rust-lang/rust/issues/26697)

# Summary

Add some methods that already exist on slices to strings. Specifically, the
following methods should be added:

- `str::into_string`
- `String::into_boxed_str`

# Motivation

Conceptually, strings and slices are similar types. Many methods are already
shared between the two types due to their similarity. However, not all methods
are shared between the types, even though many could be. This is a little
unexpected and inconsistent. Because of that, this RFC proposes to remedy this
by adding a few methods to strings to even out these two types’ available
methods.

Specifically, it is currently very difficult to construct a `Box<str>`, while it
is fairly simple to make a `Box<[T]>` by using `Vec::into_boxed_slice`. This RFC
proposes a means of creating a `Box<str>` by converting a `String`.

# Detailed design

Add the following method to `str`, presumably as an inherent method:

- `into_string(self: Box<str>) -> String`: Returns `self` as a `String`. This is
  equivalent to `[T]`’s `into_vec`.

Add the following method to `String` as an inherent method:

- `into_boxed_str(self) -> Box<str>`: Returns `self` as a `Box<str>`,
  reallocating to cut off any excess capacity if needed. This is required to
  provide a safe means of creating `Box<str>`. This is equivalent to `Vec<T>`’s
  `into_boxed_slice`.


# Drawbacks

None, yet.

# Alternatives

- The original version of this RFC had a few extra methods:
  - `str::chunks(&self, n: usize) -> Chunks`: Returns an iterator that yields
    the *characters* (not bytes) of the string in groups of `n` at a time.
    Iterator element type: `&str`.

  - `str::windows(&self, n: usize) -> Windows`: Returns an iterator over all
    contiguous windows of character length `n`. Iterator element type: `&str`.

    This and `str::chunks` aren’t really useful without proper treatment of
    graphemes, so they were removed from the RFC.

  - `<[T]>::subslice_offset(&self, inner: &[T]) -> usize`: Returns the offset
    (in elements) of an inner slice relative to an outer slice. Panics of
    `inner` is not contained within `self`.

    `str::subslice_offset` isn’t yet stable and its usefulness is dubious, so
    this method was removed from the RFC.


# Unresolved questions

None.

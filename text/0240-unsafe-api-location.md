- Start Date: 2014-10-07
- RFC PR: [rust-lang/rfcs#240](https://github.com/rust-lang/rfcs/pull/240)
- Rust Issue: [rust-lang/rust#17863](https://github.com/rust-lang/rust/issues/17863)

# Summary

This is a *conventions RFC* for settling the location of `unsafe` APIs relative
to the types they work with, as well as the use of `raw` submodules.

The brief summary is:

* Unsafe APIs should be made into methods or static functions in the same cases
  that safe APIs would be.

* `raw` submodules should be used only to *define* explicit low-level
  representations.

# Motivation

Many data structures provide unsafe APIs either for avoiding checks or working
directly with their (otherwise private) representation. For example, `string`
provides:

* An `as_mut_vec` method on `String` that provides a `Vec<u8>` view of the
  string.  This method makes it easy to work with the byte-based representation
  of the string, but thereby also allows violation of the utf8 guarantee.

* A `raw` submodule with a number of free functions, like `from_parts`, that
  constructs a `String` instances from a raw-pointer-based representation, a
  `from_utf8` variant that does not actually check for utf8 validity, and so
  on. The unifying theme is that all of these functions avoid checking some key
  invariant.

The problem is that currently, there is no clear/consistent guideline about
which of these APIs should live as methods/static functions associated with a
type, and which should live in a `raw` submodule. Both forms appear throughout
the standard library.

# Detailed design

The proposed convention is:

* When an unsafe function/method is clearly "about" a certain type (as a way of
  constructing, destructuring, or modifying values of that type), it should be a
  method or static function on that type. This is the same as the convention for
  placement of safe functions/methods. So functions like
  `string::raw::from_parts` would become static functions on `String`.

* `raw` submodules should only be used to *define* low-level
  types/representations (and methods/functions on them). Methods for converting
  to/from such low-level types should be available directly on the high-level
  types. Examples: `core::raw`, `sync::raw`.

The benefits are:

* *Ergonomics*. You can gain easy access to unsafe APIs merely by having a value
  of the type (or, for static functions, importing the type).

* *Consistency and simplicity*. The rules for placement of unsafe APIs are the
  same as those for safe APIs.

The perspective here is that marking APIs `unsafe` is enough to deter their use
in ordinary situations; they don't need to be further distinguished by placement
into a separate module.

There are also some naming conventions to go along with unsafe static functions
and methods:

* When an unsafe function/method is an unchecked variant of an otherwise safe
  API, it should be marked using an `_unchecked` suffix.

  For example, the `String` module should provide both `from_utf8` and
  `from_utf8_unchecked` constructors, where the latter does not actually check
  the utf8 encoding.  The `string::raw::slice_bytes` and
  `string::raw::slice_unchecked` functions should be merged into a single
  `slice_unchecked` method on strings that checks neither bounds nor utf8
  boundaries.

* When an unsafe function/method produces or consumes a low-level representation
  of a data structure, the API should use `raw` in its name. Specifically,
  `from_raw_parts` is the typical name used for constructing a value from e.g. a
  pointer-based representation.

* Otherwise, *consider* using a name that suggests *why* the API is unsafe. In
  some cases, like `String::as_mut_vec`, other stronger conventions apply, and the
  `unsafe` qualifier on the signature (together with API documentation) is
  enough.

The unsafe methods and static functions for a given type should be placed in
their own `impl` block, at the end of the module defining the type; this will
ensure that they are grouped together in rustdoc. (Thanks @kballard for the
suggestion.)

# Drawbacks

One potential drawback of these conventions is that the documentation for a
module will be cluttered with rarely-used `unsafe` APIs, whereas the `raw`
submodule approach neatly groups these APIs.  But rustdoc could easily be
changed to either hide or separate out `unsafe` APIs by default, and in the
meantime the `impl` block grouping should help.

More specifically, the convention of placing unsafe constructors in `raw` makes
them very easy to find. But the usual `from_` convention, together with the
naming conventions suggested above, should make it fairly easy to discover such
constructors even when they're supplied directly as static functions.

More generally, these conventions give `unsafe` APIs more equal status with safe
APIs. Whether this is a *drawback* depends on your philosophy about the status
of unsafe programming. But on a technical level, the key point is that the APIs
are marked `unsafe`, so users still have to opt-in to using them. *Ed note: from
my perspective, low-level/unsafe programming is important to support, and there
is no reason to penalize its ergonomics given that it's opt-in anyway.*

# Alternatives

There are a few alternatives:

* Rather than providing unsafe APIs directly as methods/static functions, they
  could be grouped into a single extension trait. For example, the `String` type
  could be accompanied by a `StringRaw` extension trait providing APIs for
  working with raw string representations. This would allow a clear grouping of
  unsafe APIs, while still providing them as methods/static functions and
  allowing them to easily be imported with e.g. `use std::string::StringRaw`.
  On the other hand, it still further penalizes the raw APIs (beyond marking
  them `unsafe`), and given that rustdoc could easily provide API grouping, it's
  unclear exactly what the benefit is.

* ([Suggested by @kballard](https://github.com/rust-lang/rfcs/pull/240#issuecomment-55635468)):

  > Use `raw` for functions that construct a value of the type without checking
  > for one or more invariants.

  The advantage is that it's easy to find such invariant-ignoring functions. The
  disadvantage is that their ergonomics is worsened, since they much be
  separately imported or referenced through a lengthy path:

  ```rust
  // Compare the ergonomics:
  string::raw::slice_unchecked(some_string, start, end)
  some_string.slice_unchecked(start, end)
  ```

* Another suggestion by @kballard is to keep the basic structure of `raw`
  submodules, but use associated types to improve the ergonomics. Details (and
  discussions of pros/cons) are in
  [this comment](https://github.com/rust-lang/rfcs/pull/240/files#r17572875).

* Use `raw` submodules to group together *all* manipulation of low-level
  representations. No module in `std` currently does this; existing modules
  provide some free functions in `raw`, and some unsafe methods, without a clear
  driving principle. The ergonomics of moving *everything* into free functions
  in a `raw` submodule are quite poor.

# Unresolved questions

The `core::raw` module provides structs with public representations equivalent
to several built-in and library types (boxes, closures, slices, etc.). It's not
clear whether the name of this module, or the location of its contents, should
change as a result of this RFC. The module is a special case, because not all of
the types it deals with even have corresponding modules/type declarations -- so
it probably suffices to leave decisions about it to the API stabilization
process.

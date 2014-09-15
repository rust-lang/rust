- Start Date: (fill me in with today's date, 2014-09-15)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This is a *conventions RFC* for settling the location of `unsafe` APIs relative
to the types they work with, as well as the use of `raw` submodules.

The brief summary is:

* Unsafe APIs should be made into methods or static functions in the same cases
  that safe APIs would be.

* `raw` submodules should be used only to provide APIs directly on low-level
  representations.

# Motivation

Many data structures provide unsafe APIs either for avoiding checks or working
directly with their (otherwise private) representation. For example, `string`
provides:

* An `as_mut_vec` method on `String` that provides a `Vec<u8>` view of the
  string.  This method makes it easy to work with the byte-based representation
  of the string, but thereby also allows violation of the utf8 guarantee.

* A `raw` submodule with a number of free functions, like `from_parts`, that
  construct `String` instances from various raw-pointer-based representations,
  as well as a `from_utf8` variant that does not actually check for utf8
  validity.

The problem is that currently, there is no consistent guideline about which of
these APIs should live as methods/static functions associated with a type, and
which should live in a `raw` submodule. Both forms appear throughout the
standard library.

# Detailed design

The proposed convention is:

* When an unsafe function/method is clearly "about" a certain type (as a way of
  constructing, destructuring, or modifying values of that type), it should be a
  method or static function on that type. This is the same as the convention for
  placement of safe functions/methods.

* When an unsafe function/method is specifically for producing or consuming the
  underlying representation of a data structure (which is otherwise private),
  the API should use `raw` in its name. Specifically, `from_raw_parts` is the
  typical name used for constructing a value from e.g. a pointer-based
  representation.

* When an unsafe function/method is an unchecked variant of an otherwise safe
  API, it should be marked using an `_unchecked` suffix.

* `raw` submodules should only be used to provide APIs directly on low-level
  representations, separately from functions/methods for converting to/from such
  raw representations.

The benefit to moving unsafe APIs into methods (resp. static functions) is the
usual one: you can gain easy access to these APIs merely by having a value of
the type (resp. importing the type).

The perspective here is that marking APIs `unsafe` is enough to deter their use
in ordinary situations; they don't need to be further distinguished by placement
into a separate module.

# Drawbacks

One potential drawback of these conventions is that the documentation for a
module will be cluttered with rarely-used `unsafe` APIs, whereas the `raw`
submodule approach neatly groups these APIs.  But rustdoc could easily be
changed to either hide or separate out `unsafe` APIs by default.

More generally, these conventions give `unsafe` APIs more equal status with safe
APIs. Whether this is a *drawback* depends on your philosophy about the status
of unsafe programming. But on a technical level, the key point is that the APIs
are marked `unsafe`, so users still have to opt-in to using them. *Ed note: from
my perspective, low-level/unsafe programming is important to support, and there
is no reason to penalize its ergonomics given that it's opt-in anyway.*

# Alternatives

There are two main alternatives:

* Rather than providing unsafe APIs directly as methods/static functions, they
  could be grouped into a single extension trait. For example, the `String` type
  could be accompanied by a `StringRaw` extension trait providing APIs for
  working with raw string representations. This would allow a clear grouping of
  unsafe APIs, while still providing them as methods/static functions and
  allowing them to easily be imported with e.g. `use std::string::StringRaw`.
  On the other hand, it still further penalizes the raw APIs (beyond marking
  them `unsafe`), and given that rustdoc could easily provide API grouping, it's
  unclear exactly what the benefit is.

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

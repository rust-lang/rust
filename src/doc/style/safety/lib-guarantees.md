% Library-level guarantees

Most libraries rely on internal invariants, e.g. about their data, resource
ownership, or protocol states. In Rust, broken invariants cannot produce
segfaults, but they can still lead to wrong answers.

### Provide library-level guarantees whenever practical. **[FIXME: needs RFC]**

Library-level invariants should be turned into guarantees whenever
practical. They should hold no matter what the client does, modulo
explicit opt-outs. Depending on the kind of invariant, this can be
achieved through a combination of static and dynamic enforcement, as
described below.

#### Static enforcement:

Guaranteeing invariants almost always requires _hiding_,
i.e. preventing the client from directly accessing or modifying
internal data.

For example, the representation of the `str` type is hidden,
which means that any value of type `str` must have been produced
through an API under the control of the `str` module, and these
APIs in turn ensure valid utf-8 encoding.

Rust's type system makes it possible to provide guarantees even while
revealing more of the representation than usual. For example, the
`as_bytes()` method on `&str` gives a _read-only_ view into the
underlying buffer, which cannot be used to violate the utf-8 property.

#### Dynamic enforcement:

Malformed inputs from the client are hazards to library-level
guarantees, so library APIs should validate their input.

For example, `std::str::from_utf8_owned` attempts to convert a `u8`
slice into an owned string, but dynamically checks that the slice is
valid utf-8 and returns `Err` if not.

See
[the discussion on input validation](../features/functions-and-methods/input.md)
for more detail.


### Prefer static enforcement of guarantees. **[FIXME: needs RFC]**

Static enforcement provides two strong benefits over dynamic enforcement:

* Bugs are caught at compile time.
* There is no runtime cost.

Sometimes purely static enforcement is impossible or impractical. In these
cases, a library should check as much as possible statically, but defer to
dynamic checks where needed.

For example, the `std::string` module exports a `String` type with the guarantee
that all instances are valid utf-8:

* Any _consumer_ of a `String` is statically guaranteed utf-8 contents. For example,
  the `append` method can push a `&str` onto the end of a `String` without
  checking anything dynamically, since the existing `String` and `&str` are
  statically guaranteed to be in utf-8.

* Some _producers_ of a `String` must perform dynamic checks. For example, the
  `from_utf8` function attempts to convert a `Vec<u8>` into a `String`, but
  dynamically checks that the contents are utf-8.

### Provide opt-outs with caution; make them explicit. **[FIXME: needs RFC]**

Providing library-level guarantees sometimes entails inconvenience (for static
checks) or overhead (for dynamic checks). So it is sometimes desirable to allow
clients to sidestep this checking, while promising to use the API in a way that
still provides the guarantee. Such escape hatches should only be introduced when
there is a demonstrated need for them.

It should be trivial for clients to audit their use of the library for
escape hatches.

See
[the discussion on input validation](../features/functions-and-methods/input.md)
for conventions on marking opt-out functions.

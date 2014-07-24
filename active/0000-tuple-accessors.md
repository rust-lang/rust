- Start Date: 2014-07-24
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

Summary
=======

Add simple syntax for accessing values within tuples and tuple structs.

Motivation
==========

Right now accessing fields of tuples and tuple structs is incredibly painful—one
must rely on pattern matching alone to extract values. This became such a
problem that twelve traits were created in the standard library
(`core::tuple::Tuple*`) to make tuple value accesses easier, adding `.valN()`,
`.refN()`, and `.mutN()` methods to help this. But this is not a very nice
solution—it requires the traits to be implemented in the standard library, not
the language, and for those traits to be imported on use. On the whole this is
not a problem, because most of the time `std::prelude::*` is imported, but this
is still a hack which is not a real solution to the problem at hand. It also
only supports tuples of length up to twelve, which is normally not a problem but
emphasises how bad the current situation is.

Detailed design
===============

Add syntax of the form `<expr>.<integer>` for accessing values within tuples and
tuple structs. This syntax is recognised wherever an integer or float literal is
found in place of the normal field or method name expected when accessing fields
with `.`. Float literals in this position are expanded into two field accesses,
so that an expression of the form `a.1.3` is equivalent to `(a.1).3`.

Accessing a tuple or tuple struct field like so:

```rust
let x = (box 1i, box 2i);
let x1 = x.1;
```

is roughly equivalent to:

```rust
let x = (box 1i, box 2i);
let x1 = { let (_, a) = x; a };
```

However, when taking a (possibly mutable) reference to a field, the equivalent
expansion is slightly different:

```rust
let x = (box 1i, box 2i);
let x1 = &x.1;
```

is roughly equivalent to:

```rust
let x = (box 1i, box 2i);
let x1 = { let (_, ref a) = x; a };
```

A similar process is performed with `&mut`.

Drawbacks
=========

More complexity that is not strictly necessary.

Alternatives
============

Allow indexing of tuples and tuple structs: this has the advantage of
consistency, but the disadvantage of not being checked for out-of-bounds errors
at compile time.

Unresolved questions
====================

None.

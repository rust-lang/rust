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
tuple structs.  This syntax is recognised wherever an unsuffixed integer or
float literal is found in place of the normal field or method name expected when
accessing fields with `.`. Float literals in this position are expanded into two
field accesses, so that an expression of the form `a.1.3` is equivalent to
`(a.1).3`.

Tuple/tuple struct field access behaves the same way as accessing named fields
on normal structs:

```rust
// With tuple struct
struct Foo(int, int);
let mut foo = Foo(3, -15);
foo.0 = 5;
assert_eq!(foo.0, 5);

// With normal struct
struct Foo2 { _0: int, _1: int }
let mut foo2 = Foo2 { _0: 3, _1: -15 };
foo2._0 = 5;
assert_eq!(foo2._0, 5);
```

Effectively, a tuple or tuple struct field is just a normal named field with an
integer for a name.

Drawbacks
=========

More complexity that is not strictly necessary.

Alternatives
============

Stay with the status quo. Either recommend using a struct with named fields or
suggest using pattern matching to extract values. If extracting individual
fields of tuples is really necessary, `the `TupleN` traits could be used
instead, and something like `#[deriving(Tuple3)]` could possibly be added for
tuple structs.

Unresolved questions
====================

None.

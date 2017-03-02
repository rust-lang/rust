- Start Date: 2014-11-29
- RFC PR: [490](https://github.com/rust-lang/rfcs/pull/490)
- Rust Issue: [19607](https://github.com/rust-lang/rust/issues/19607)

Summary
=======

Change the syntax for dynamically sized type parameters from `Sized? T` to `T:
?Sized`, and change the syntax for traits for dynamically sized types to `trait
Foo for ?Sized`. Extend this new syntax to work with `where` clauses.

Motivation
==========

History of the DST syntax
-------------------------

When dynamically sized types were first designed, and even when they were first
being implemented, the syntax for dynamically sized type parameters had not been
fully settled on. Initially, dynamically sized type parameters were denoted by a
leading `unsized` keyword:

```rust
fn foo<unsized T>(x: &T) { ... }
struct Foo<unsized T> { field: T }
// etc.
```

This is the syntax used in Niko Matsakis’s [initial design for
DST](http://smallcultfollowing.com/babysteps/blog/2014/01/05/dst-take-5/). This
syntax makes sense to those who are familiar with DST, but has some issues which
could be perceived as problems for those learning to work with dynamically sized
types:

- It implies that the parameter *must* be unsized, where really it’s only
  optional;
- It does not visually relate to the `Sized` trait, which is fundamentally
  related to declaring a type as unsized (removing the default `Sized` bound).

Later, Felix S. Klock II [came up with an alternative
syntax](http://blog.pnkfx.org/blog/2014/03/13/an-insight-regarding-dst-grammar-for-rust/)
using the `type` keyword:

```rust
fn foo<type T>(x: &T) { ... }
struct Foo<type T> { field: T }
// etc.
```

The inspiration behind this is that the union of all sized types and all unsized
types is simply all types. Thus, it makes sense for the most general type
parameter to be written as `type T`.

This syntax resolves the first problem listed above (i.e., it no longer implies
that the type *must* be unsized), but does not resolve the second. Additionally,
it is possible that some people could be confused by the use of the `type`
keyword, as it contains little meaning—one would assume a bare `T` as a *type*
parameter to be a type already, so what does adding a `type` keyword mean?

Perhaps because of these concerns, the syntax for dynamically sized type
parameters has since been changed one more time, this time to use the `Sized`
trait’s name followed by a question mark:

```rust
fn foo<Sized? T>(x: &T) { ... }
struct Foo<Sized? T> { field: T }
// etc.
```

This syntax simply removes the implicit `Sized` bound on every type parameter
using the `?` symbol. It resolves the problem about not mentioning `Sized` that
the first two syntaxes didn’t. It also hints towards being related to sizedness,
resolving the problem that plagued `type`. It also successfully states that
unsizedness is only *optional*—that the parameter may be sized or unsized. This
syntax has stuck, and is the syntax used today. Additionally, it could
potentially be extended to other traits: for example, a new pointer type that
cannot be dropped, `&uninit`, could be added, requiring that it be written to
before being dropped.  However, many generic functions assume that any parameter
passed to them can be dropped. `Drop` could be made a default bound to resolve
this, and `Drop?` would remove this bound from a type parameter.

The problem with `Sized? T`
---------------------------

There is some inconsistency present with the `Sized` syntax. After going through
multiple syntaxes for DST, all of which were keywords preceding type parameters,
the `Sized?` annotation stayed *before* the type parameter’s name when it was
adopted as the syntax for dynamically sized type parameters. This can be
considered inconsistent in some ways—`Sized?` looks like a bound, contains a
trait name like a bound does, and changes what types can unify with the type
parameter like a bound does, but does not come *after* the type parameter’s name
like a bound does. This also is inconsistent with Rust’s general pattern of not
using C-style variable declarations (`int x`) but instead using a colon and
placing the type after the name (`x: int`). (A type parameter is not strictly a
variable declaration, but is similar: it declares a new name in a scope.) These
problems together make `Sized?` the only marker that comes before type parameter
or even variable names, and with the addition of negative bounds, it looks even
more inconsistent:

```rust
// Normal bound
fn foo<T: Foo>() {}
// Negative bound
fn foo<T: !Foo>() {}
// Generalising ‘anti-bound’
fn foo<Foo? T>() {}
```

The syntax also looks rather strange when recent features like associated types
and `where` clauses are considered:

```rust
// This `where` clause syntax doesn’t work today, but perhaps should:
trait Foo<T> where Sized? T {
    type Sized? Bar;
}
```

Furthermore, the `?` on `Sized?` comes after the trait name, whereas most
unary-operator-like symbols in the Rust language come before what they are
attached to.

This RFC proposes to change the syntax for dynamically sized type parameters to
`T: ?Sized` to resolve these issues.

Detailed design
===============

Change the syntax for dynamically sized type parameters to `T: ?Sized`:

```rust
fn foo<T: ?Sized>(x: &T) { ... }
struct Foo<T: Send + ?Sized + Sync> { field: Box<T> }
trait Bar { type Baz: ?Sized; }
// etc.
```

Change the syntax for traits for dynamically-sized types to have a prefix `?`
instead of a postfix one:

```rust
trait Foo for ?Sized { ... }
```

Allow using this syntax in `where` clauses:

```rust
fn foo<T>(x: &T) where T: ?Sized { ... }
```

Drawbacks
=========

- The current syntax uses position to distinguish between removing and adding
  bounds, while the proposed syntax only uses a symbol. Since `?Sized` is
  actually an anti-bound (it removes a bound), it (in some ways) makes sense to
  put it on the opposite side of a type parameter to show this.

- Only a single character separates adding a `Sized` bound and removing an
  implicit one. This shouldn’t be a problem in general, as adding a `Sized`
  bound to a type parameter is pointless (because it is implicitly there
  already). A lint could be added to check for explicit default bounds if this
  turns out to be a problem.

Alternatives
============

- Choose one of the previous syntaxes or a new syntax altogether. The drawbacks
  of the previous syntaxes are discussed in the ‘History of the DST syntax’
  section of this RFC.

- Change the syntax to `T: Sized?` instead. This is less consistent with things
  like negative bounds (which would probably be something like `T: !Foo`), and
  uses a suffix operator, which is less consistent with other parts of Rust’s
  syntax. It is, however, closer to the current syntax (`Sized? T`), and looks
  more natural because of how `?` is used in natural languages such as English.

Unresolved questions
====================

None.

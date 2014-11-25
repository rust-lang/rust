- Start Date: 2014-11-05
- RFC PR: [rust-lang/rfcs#445](https://github.com/rust-lang/rfcs/pull/445)
- Rust Issue: [rust-lang/rust#19324](https://github.com/rust-lang/rust/issues/19324)

# Summary

This is a conventions RFC establishing a definition and naming
convention for *extension traits*: `FooExt`.

# Motivation

This RFC is part of the ongoing API conventions and stabilization
effort.

Extension traits are a programming pattern that makes it
possible to add methods to an existing type outside of the crate
defining that type. While they should be used sparingly, the new
[object safety rules](https://github.com/rust-lang/rfcs/pull/255) have
increased the need for this kind of trait, and hence the need for a
clear convention.

# Detailed design

## What is an extension trait?

Rust currently allows inherent methods to be defined on a type only in
the crate where that type is defined. But it is often the case that
clients of a type would like to incorporate additional methods to
it. Extension traits are a pattern for doing so:

```rust
extern crate foo;
use foo::Foo;

trait FooExt {
    fn bar(&self);
}

impl FooExt for Foo {
    fn bar(&self) { .. }
}
```

By defining a new trait, a client of `foo` can add new methods to `Foo`.

Of course, adding methods via a new trait happens all the time. What
makes it an *extension* trait is that the trait is not designed for
*generic* use, but only as way of adding methods to a specific type or
family of types.

This is of course a somewhat subjective distinction. Whenever
designing an extension trait, one should consider whether the trait
could be used in some more generic way. If so, the trait should be
named and exported as if it were just a "normal" trait. But traits
offering groups of methods that really only make sense in the context
of some particular type(s) are true extension traits.

The new
[object safety rules](https://github.com/rust-lang/rfcs/pull/255) mean
that a trait can only be used for trait objects if *all* of its
methods are usable; put differently, it ensures that for "object safe
traits" there is always a canonical way to implement `Trait` for
`Box<Trait>`. To deal with this new rule, it is sometimes necessary to
break traits apart into an object safe trait and extension traits:

```rust
// The core, object-safe trait
trait Iterator<A> {
    fn next(&mut self) -> Option<A>;
}

// The extension trait offering object-unsafe methods
trait IteratorExt<A>: Iterator<A> {
    fn chain<U: Iterator<A>>(self, other: U) -> Chain<Self, U> { ... }
    fn zip<B, U: Iterator<B>>(self, other: U) -> Zip<Self, U> { ... }
    fn map<B>(self, f: |A| -> B) -> Map<'r, A, B, Self> { ... }
    ...
}

// A blanket impl
impl<A, I> IteratorExt<A> for I where I: Iterator<A> {
    ...
}
```

Note that, although this split-up definition is somewhat more complex,
it is also more flexible: because `Box<Iterator<A>>` will implement
`Iterator<A>`, you can now use *all* of the adapter methods provided
in `IteratorExt` on trait objects, even though they are not object
safe.

## The convention

The proposed convention is, first of all, to (1) prefer adding default
methods to existing traits or (2) prefer generically useful traits to
extension traits whenever feasible.

For true extension traits, there should be a clear type or trait that
they are extending. The extension trait should be called `FooExt`
where `Foo` is that type or trait.

In some cases, the extension trait only applies conditionally. For
example, `AdditiveIterator` is an extension trait currently in `std`
that applies to iterators over numeric types. These extension traits
should follow a similar convention, putting together the type/trait
name and the qualifications, together with the `Ext` suffix:
`IteratorAddExt`.

### What about `Prelude`?

A [previous convention](https://github.com/rust-lang/rfcs/pull/344)
used a `Prelude` suffix for extension traits that were also part of
the `std` prelude; this new convention deprecates that one.

## Future proofing

In the future, the need for many of these extension traits may
disappear as other languages features are added. For example,
method-level `where` clauses will eliminate the need for
`AdditiveIterator`. And allowing inherent `impl`s like `impl<T: Trait>
T { .. }` for the crate defining `Trait` would eliminate even more.

However, there will always be *some* use of extension traits, and we
need to stabilize the 1.0 libraries prior to these language features
landing. So this is the proposed convention for now, and in the future
it may be possible to deprecate some of the resulting traits.

# Alternatives

It seems clear that we need *some* convention here. Other possible
suffixes would be `Util` or `Methods`, but `Ext` is both shorter and
connects to the name of the pattern.

# Drawbacks

In general, extension traits tend to require additional imports --
especially painful when dealing with object safety. However, this is
more to do with the language as it stands today than with the
conventions in this RFC.

Libraries are already starting to export their own `prelude` module
containing extension traits among other things, which by convention is
glob imported.

In the long run, we should add a general "prelude" facility for
external libraries that makes it possible to *globally* import a small
set of names from the crate. Some early investigations of such a
feature are already under way, but are outside the scope of this RFC.

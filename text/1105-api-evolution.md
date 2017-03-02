- Feature Name: not applicable
- Start Date: 2015-05-04
- RFC PR: [rust-lang/rfcs#1105](https://github.com/rust-lang/rfcs/pull/1105)
- Rust Issue: N/A

# Summary

This RFC proposes a comprehensive set of guidelines for which changes to
*stable* APIs are considered breaking from a semver perspective, and which are
not.  These guidelines are intended for both the standard library and for the
crates.io ecosystem.

This does *not* mean that the standard library should be completely free to make
non-semver-breaking changes; there are sometimes still risks of ecosystem pain
that need to be taken into account. Rather, this RFC makes explicit an initial
set of changes that absolutely *cannot* be made without a semver bump.

Along the way, it also discusses some interactions with potential language
features that can help mitigate pain for non-breaking changes.

The RFC covers only API issues; other issues related to language features,
lints, type inference, command line arguments, Cargo, and so on are considered
out of scope.

# Motivation

Both Rust and its library ecosystem have adopted [semver](http://semver.org/), a
technique for versioning platforms/libraries partly in terms of the effect on
the code that uses them. In a nutshell, the versioning scheme has three components::

1. **Major**: must be incremented for changes that break client code.
2. **Minor**: incremented for backwards-compatible feature additions.
3. **Patch**: incremented for backwards-compatible bug fixes.

[Rust 1.0.0](http://blog.rust-lang.org/2015/02/13/Final-1.0-timeline.html) will
mark the beginning of our
[commitment to stability](http://blog.rust-lang.org/2014/10/30/Stability.html),
and from that point onward it will be important to be clear about what
constitutes a breaking change, in order for semver to play a meaningful role. As
we will see, this question is more subtle than one might think at first -- and
the simplest approach would make it effectively impossible to grow the standard
library.

The goal of this RFC is to lay out a comprehensive policy for what *must* be
considered a breaking API change from the perspective of semver, along with some
guidance about non-semver-breaking changes.

# Detailed design

For clarity, in the rest of the RFC, we will use the following terms:

* **Major change**: a change that requires a major semver bump.
* **Minor change**: a change that requires only a minor semver bump.
* **Breaking change**: a change that, *strictly speaking*, can cause downstream
  code to fail to compile.

What we will see is that in Rust today, almost any change is technically a
breaking change. For example, given the way that globs currently work, *adding
any public item* to a library can break its clients (more on that later). But
not all breaking changes are equal.

So, this RFC proposes that **all major changes are breaking, but not all breaking
changes are major.**

## Overview

### Principles of the policy

The basic design of the policy is that **the same code should be able to run
against different minor revisions**. Furthermore, minor changes should require
at most a few local *annotations* to the code you are developing, and in
principle no changes to your dependencies.

In more detail:

* Minor changes should require at most minor amounts of work upon upgrade. For
  example, changes that may require occasional type annotations or use of UFCS
  to disambiguate are not automatically "major" changes. (But in such cases, one
  must evaluate how widespread these "minor" changes are).

* In principle, it should be possible to produce a version of dependency code
  that *will not break* when upgrading other dependencies, or Rust itself, to a
  new minor revision. This goes hand-in-hand with the above bullet; as we will
  see, it's possible to save a fully "elaborated" version of upstream code that
  does not require any disambiguation. The "in principle" refers to the fact
  that getting there may require some additional tooling or language support,
  which this RFC outlines.

That means that any breakage in a minor release must be very "shallow": it must
always be possible to locally fix the problem through some kind of
disambiguation *that could have been done in advance* (by using more explicit
forms) or other annotation (like disabling a lint). It means that minor changes
can never leave you in a state that requires breaking changes to your own code.

**Although this general policy allows some (very limited) breakage in minor
releases, it is not a license to make these changes blindly**. The breakage that
this RFC permits, aside from being very simple to fix, is also unlikely to occur
often in practice. The RFC will discuss measures that should be employed in the
standard library to ensure that even these minor forms of breakage do not cause
widespread pain in the ecosystem.

### Scope of the policy

The policy laid out by this RFC applies to *stable*, *public* APIs in the
standard library. Eventually, stability attributes will be usable in external
libraries as well (this will require some design work), but for now public APIs
in external crates should be understood as de facto stable after the library
reaches 1.0.0 (per semver).

## Policy by language feature

Most of the policy is simplest to lay out with reference to specific language
features and the way that APIs using them can, and cannot, evolve in a minor
release.

**Breaking changes are assumed to be major changes unless otherwise stated**.
The RFC covers many, but not all breaking changes that are major; it covers
*all* breaking changes that are considered minor.

### Crates

#### Major change: going from stable to nightly

Changing a crate from working on stable Rust to *requiring* a nightly is
considered a breaking change. That includes using `#[feature]` directly, or
using a dependency that does so. Crate authors should consider using Cargo
["features"](http://doc.crates.io/manifest.html#the-[features]-section) for
their crate to make such use opt-in.

#### Minor change: altering the use of Cargo features

Cargo packages can provide
[opt-in features](http://doc.crates.io/manifest.html#the-[features]-section),
which enable `#[cfg]` options. When a common dependency is compiled, it is done
so with the *union* of all features opted into by any packages using the
dependency. That means that adding or removing a feature could technically break
other, unrelated code.

However, such breakage always represents a bug: packages are supposed to support
any combination of features, and if another client of the package depends on a
given feature, that client should specify the opt-in themselves.

### Modules

#### Major change: renaming/moving/removing any public items.

Although renaming an item might seem like a minor change, according to the
general policy design this is not a permitted form of breakage: it's not
possible to annotate code in advance to avoid the breakage, nor is it possible
to prevent the breakage from affecting dependencies.

Of course, much of the effect of renaming/moving/removing can be achieved by
instead using deprecation and `pub use`, and the standard library should not be
afraid to do so! In the long run, we should consider hiding at least some old
deprecated items from the docs, and could even consider putting out a major
version solely as a kind of "garbage collection" for long-deprecated APIs.

#### Minor change: adding new public items.

Note that adding new public items is currently a breaking change, due to glob
imports. For example, the following snippet of code will break if the `foo`
module introduces a public item called `bar`:

```rust
use foo::*;
fn bar() { ... }
```

The problem here is that glob imports currently do not allow any of their
imports to be shadowed by an explicitly-defined item.

This is considered a minor change because under the principles of this RFC: the
glob imports could have been written as more explicit (expanded) `use`
statements. It is also plausible to do this expansion automatically for a
crate's dependencies, to prevent breakage in the first place.

(This RFC also suggests permitting shadowing of a glob import by any explicit
item. This has been the intended semantics of globs, but has not been
implemented. The details are left to a future RFC, however.)

### Structs

See "[Signatures in type definitions](#signatures-in-type-definitions)" for some
general remarks about changes to the actual types in a `struct` definition.

#### Major change: adding a private field when all current fields are public.

This change has the effect of making external struct literals impossible to
write, which can break code irreparably.

#### Major change: adding a public field when no private field exists.

This change retains the ability to use struct literals, but it breaks existing
uses of such literals; it likewise breaks exhaustive matches against the struct.

#### Minor change: adding or removing private fields when at least one already exists (before and after the change).

No existing code could be relying on struct literals for the struct, nor on
exhaustively matching its contents, and client code will likewise be oblivious
to the addition of further private fields.

For tuple structs, this is only a minor change if furthermore *all* fields are
currently private. (Tuple structs with mixtures of public and private fields are
bad practice in any case.)

#### Minor change: going from a tuple struct with all private fields (with at least one field) to a normal struct, or vice versa.

This is technically a breaking change:

```rust
// in some other module:
pub struct Foo(SomeType);

// in downstream code
let Foo(_) = foo;
```

Changing `Foo` to a normal struct can break code that matches on it -- but there
is never any real reason to match on it in that circumstance, since you cannot
extract any fields or learn anything of interest about the struct.

### Enums

See "[Signatures in type definitions](#signatures-in-type-definitions)" for some
general remarks about changes to the actual types in an `enum` definition.

#### Major change: adding new variants.

Exhaustiveness checking means that a `match` that explicitly checks all the
variants for an `enum` will break if a new variant is added. It is not currently
possible to defend against this breakage in advance.

A [postponed RFC](https://github.com/rust-lang/rfcs/pull/757) discusses a
language feature that allows an enum to be marked as "extensible", which
modifies the way that exhaustiveness checking is done and would make it possible
to extend the enum without breakage.

#### Major change: adding new fields to a variant.

If the enum is public, so is the full contents of all of its variants. As per
the rules for structs, this means it is not allowed to add any new fields (which
will automatically be public).

If you wish to allow for this kind of extensibility, consider introducing a new,
explicit struct for the variant up front.

### Traits

#### Major change: adding a non-defaulted item.

Adding any item without a default will immediately break all trait implementations.

It's possible that in the future we will allow some kind of
"[sealing](#thoughts-on-possible-language-changes-unofficial)" to say that a trait can only be used as a bound, not
to provide new implementations; such a trait *would* allow arbitrary items to be
added.

#### Major change: any non-trivial change to item signatures.

Because traits have both implementors and consumers, any change to the signature
of e.g. a method will affect at least one of the two parties. So, for example,
abstracting a concrete method to use generics instead might work fine for
clients of the trait, but would break existing implementors. (Note, as above,
the potential for "sealed" traits to alter this dynamic.)

#### Minor change: adding a defaulted item.

Adding a defaulted item is technically a breaking change:

```rust
trait Trait1 {}
trait Trait2 {
    fn foo(&self);
}

fn use_both<T: Trait1 + Trait2>(t: &T) {
    t.foo()
}
```

If a `foo` method is added to `Trait1`, even with a default, it would cause a
dispatch ambiguity in `use_both`, since the call to `foo` could be referring to
either trait.

(Note, however, that existing *implementations* of the trait are fine.)

According to the basic principles of this RFC, such a change is minor: it is
always possible to annotate the call `t.foo()` to be more explicit *in advance*
using UFCS: `Trait2::foo(t)`. This kind of annotation could be done
automatically for code in dependencies (see
[Elaborated source](#elaborated-source)). And it would also be possible to
mitigate this problem by allowing
[method renaming on trait import](#trait-item-renaming).

While the scenario of adding a defaulted method to a trait may seem somewhat
obscure, the exact same hazards arise with *implementing existing traits* (see
below), which is clearly vital to allow; we apply a similar policy to both.

All that said, it is incumbent on library authors to ensure that such "minor"
changes are in fact minor in practice: if a conflict like `t.foo()` is likely to
arise at all often in downstream code, it would be advisable to explore a
different choice of names. More guidelines for the standard library are given
later on.

There are two circumstances when adding a defaulted item is still a major change:

* The new item would change the trait from object safe to non-object safe.
* The trait has a defaulted associated type and the item being added is a
  defaulted function/method. In this case, existing impls that override the
  associated type will break, since the function/method default will not
  apply. (See
  [the associated item RFC](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#defaults)).
* Adding a default to an existing associated type is likewise a major change if
  the trait has defaulted methods, since it will invalidate use of those
  defaults for the methods in existing trait impls.

#### Minor change: adding a defaulted type parameter.

As with "[Signatures in type definitions](#signatures-in-type-definitions)",
traits are permitted to add new type parameters as long as defaults are provided
(which is backwards compatible).

### Trait implementations

#### Major change: implementing any "fundamental" trait.

A [recent RFC](https://github.com/rust-lang/rfcs/pull/1023) introduced the idea
of "fundamental" traits which are so basic that *not* implementing such a trait
right off the bat is considered a promise that you will *never* implement the
trait. The `Sized` and `Fn` traits are examples.

The coherence rules take advantage of fundamental traits in such a way that
*adding a new implementation of a fundamental trait to an existing type can
cause downstream breakage*. Thus, such impls are considered major changes.

#### Minor change: implementing any non-fundamental trait.

Unfortunately, implementing any existing trait can cause breakage:

```rust
// Crate A
    pub trait Trait1 {
        fn foo(&self);
    }

    pub struct Foo; // does not implement Trait1

// Crate B
    use crateA::Trait1;

    trait Trait2 {
        fn foo(&self);
    }

    impl Trait2 for crateA::Foo { .. }

    fn use_foo(f: &crateA::Foo) {
        f.foo()
    }
```

If crate A adds an implementation of `Trait1` for `Foo`, the call to `f.foo()`
in crate B will yield a dispatch ambiguity (much like the one we saw for
defaulted items). Thus *technically implementing any existing trait is a
breaking change!* Completely prohibiting such a change is clearly a non-starter.

However, as before, this kind of breakage is considered "minor" by the
principles of this RFC (see "Adding a defaulted item" above).

### Inherent implementations

#### Minor change: adding any inherent items.

Adding an inherent item cannot lead to dispatch ambiguity, because inherent
items trump any trait items with the same name.

However, introducing an inherent item *can* lead to breakage if the signature of
the item does not match that of an in scope, implemented trait:

```rust
// Crate A
    pub struct Foo;

// Crate B
    trait Trait {
        fn foo(&self);
    }

    impl Trait for crateA::Foo { .. }

    fn use_foo(f: &crateA::Foo) {
        f.foo()
   }
```

If crate A adds a method:

```rust
impl Foo {
    fn foo(&self, x: u8) { ... }
}
```

then crate B would no longer compile, since dispatch would prefer the inherent
impl, which has the wrong type.

Once more, this is considered a minor change, since UFCS can disambiguate (see
"Adding a defaulted item" above).

It's worth noting, however, that if the signatures *did* happen to match then
the change would no longer cause a compilation error, but might silently change
runtime behavior. The case where the same method for the same type has
meaningfully different behavior is considered unlikely enough that the RFC is
willing to permit it to be labeled as a minor change -- and otherwise, inherent
methods could never be added after the fact.

### Other items

Most remaining items do not have any particularly unique items:

* For type aliases, see "[Signatures in type definitions](#signatures-in-type-definitions)".
* For free functions, see "[Signatures in functions](#signatures-in-functions)".

## Cross-cutting concerns

### Behavioral changes

This RFC is largely focused on API changes which may, in particular, cause
downstream code to stop compiling. But in some sense it is even more pernicious
to make a change that allows downstream code to continue compiling, but causes
its runtime behavior to break.

This RFC does not attempt to provide a comprehensive policy on behavioral
changes, which would be extremely difficult. In general, APIs are expected to
provide explicit contracts for their behavior via documentation, and behavior
that is not part of this contract is permitted to change in minor
revisions. (Remember: this RFC is about setting a *minimum* bar for when major
version bumps are required.)

This policy will likely require some revision over time, to become more explicit
and perhaps lay out some best practices.

### Signatures in type definitions

#### Major change: tightening bounds.

Adding new constraints on existing type parameters is a breaking change, since
existing uses of the type definition can break. So the following is a major
change:

```rust
// MAJOR CHANGE

// Before
struct Foo<A> { .. }

// After
struct Foo<A: Clone> { .. }
```

#### Minor change: loosening bounds.

Loosening bounds, on the other hand, cannot break code because when you
reference `Foo<A>`, you *do not learn anything about the bounds on `A`*. (This
is why you have to repeat any relevant bounds in `impl` blocks for `Foo`, for
example.) So the following is a minor change:

```rust
// MINOR CHANGE

// Before
struct Foo<A: Clone> { .. }

// After
struct Foo<A> { .. }
```

#### Minor change: adding defaulted type parameters.

All existing references to a type/trait definition continue to compile and work
correctly after a new defaulted type parameter is added. So the following is
a minor change:

```rust
// MINOR CHANGE

// Before
struct Foo { .. }

// After
struct Foo<A = u8> { .. }
```

#### Minor change: generalizing to generics.

A struct or enum field can change from a concrete type to a generic type
parameter, provided that the change results in an identical type for all
existing use cases. For example, the following change is permitted:

```rust
// MINOR CHANGE

// Before
struct Foo(pub u8);

// After
struct Foo<T = u8>(pub T);
```

because existing uses of `Foo` are shorthand for `Foo<u8>` which yields the
identical field type. (Note: this is not actually true today, since
[default type parameters](https://github.com/rust-lang/rfcs/pull/213) are not
fully implemented. But this is the intended semantics.)

On the other hand, the following is not permitted:

```rust
// MAJOR CHANGE

// Before
struct Foo<T = u8>(pub T, pub u8);

// After
struct Foo<T = u8>(pub T, pub T);
```

since there may be existing uses of `Foo` with a non-default type parameter
which would break as a result of the change.

It's also permitted to change from a generic type to a more-generic one in a
minor revision:

```rust
// MINOR CHANGE

// Before
struct Foo<T>(pub T, pub T);

// After
struct Foo<T, U = T>(pub T, pub U);
```

since, again, all existing uses of the type `Foo<T>` will yield the same field
types as before.

### Signatures in functions

All of the changes mentioned below are considered major changes in the context
of trait methods, since they can break implementors.

#### Major change: adding/removing arguments.

At the moment, Rust does not provide defaulted arguments, so any change in arity
is a breaking change.

#### Minor change: introducing a new type parameter.

Technically, adding a (non-defaulted) type parameter can break code:

```rust
// MINOR CHANGE (but causes breakage)

// Before
fn foo<T>(...) { ... }

// After
fn foo<T, U>(...) { ... }
```

will break any calls like `foo::<u8>`. However, such explicit calls are rare
enough (and can usually be written in other ways) that this breakage is
considered minor. (However, one should take into account how likely it is that
the function in question is being called with explicit type arguments).  This
RFC also suggests adding a `...` notation to explicit parameter lists to keep
them open-ended (see suggested language changes).

Such changes are an important ingredient of abstracting to use generics, as
described next.

#### Minor change: generalizing to generics.

The type of an argument to a function, or its return value, can be *generalized*
to use generics, including by introducing a new type parameter (as long as it
can be instantiated to the original type). For example, the following change is
allowed:

```rust
// MINOR CHANGE

// Before
fn foo(x: u8) -> u8;
fn bar<T: Iterator<Item = u8>>(t: T);

// After
fn foo<T: Add>(x: T) -> T;
fn bar<T: IntoIterator<Item = u8>>(t: T);
```

because all existing uses are instantiations of the new signature. On the other
hand, the following isn't allowed in a minor revision:

```rust
// MAJOR CHANGE

// Before
fn foo(x: Vec<u8>);

// After
fn foo<T: Copy + IntoIterator<Item = u8>>(x: T);
```

because the generics include a constraint not satisfied by the original type.

Introducing generics in this way can potentially create type inference failures,
but these are considered acceptable per the principles of the RFC: they only
require local annotations that could have been inserted in advance.

Perhaps somewhat surprisingly, generalization applies to trait objects as well,
given that every trait implements itself:

```rust
// MINOR CHANGE

// Before
fn foo(t: &Trait);

// After
fn foo<T: Trait + ?Sized>(t: &T);
```

(The use of `?Sized` is essential; otherwise you couldn't recover the original
signature).

### Lints

#### Minor change: introducing new lint warnings/errors

Lints are considered advisory, and changes that cause downstream code to receive
additional lint warnings/errors are still considered "minor" changes.

Making this work well in practice will likely require some infrastructure work
along the lines of
[this RFC issue](https://github.com/rust-lang/rfcs/issues/1029)

## Mitigation for minor changes

### The Crater tool

@brson has been hard at work on a tool called "Crater" which can be used to
exercise changes on the entire crates.io ecosystem, looking for
regressions. This tool will be indispensable when weighing the costs of a minor
change that might cause some breakage -- we can actually gauge what the breakage
would look like in practice.

While this would, of course, miss code not available publicly, the hope is that
code on crates.io is a broadly representative sample, good enough to turn up
problems.

Any breaking, but minor change to the standard library must be evaluated through
Crater before being committed.

### Nightlies

One line of defense against a "minor" change causing significant breakage is the
nightly release channel: we can get feedback about breakage long before it makes
even into a beta release. And of course the beta cycle itself provides another
line of defense.

### Elaborated source

When compiling upstream dependencies, it is possible to generate an "elaborated"
version of the source code where all dispatch is resolved to explicit UFCS form,
all types are annotated, and all glob imports are replaced by explicit imports.

This fully-elaborated form is almost entirely immune to breakage due to any of
the "minor changes" listed above.

You could imagine Cargo storing this elaborated form for dependencies upon
compilation. That would in turn make it easy to update Rust, or some subset of
dependencies, without breaking any upstream code (even in minor ways). You would
be left only with very small, local changes to make to the code you own.

While this RFC does not propose any such tooling change right now, the point is
mainly that there are a lot of options if minor changes turn out to cause
breakage more often than anticipated.

### Trait item renaming

One very useful mechanism would be the ability to import a trait while renaming
some of its items, e.g. `use some_mod::SomeTrait with {foo_method as bar}`. In
particular, when methods happen to conflict across traits defined in separate
crates, a user of the two traits could rename one of the methods out of the way.

## Thoughts on possible language changes (unofficial)

The following is just a quick sketch of some focused language changes that would
help our API evolution story.

**Glob semantics**

As already mentioned, the fact that glob imports currently allow *no* shadowing
is deeply problematic: in a technical sense, it means that the addition of *any*
public item can break downstream code arbitrarily.

It would be much better for API evolution (and for ergonomics and intuition) if
explicitly-defined items trump glob imports. But this is left to a future RFC.

**Globs with fine-grained control**

Another useful tool for working with globs would be the ability to *exclude*
certain items from a glob import, e.g. something like:

```rust
use some_module::{* without Foo};
```

This is especially useful for the case where multiple modules being glob
imported happen to export items with the same name.

Another possibility would be to not make it an error for two glob imports to
bring the same name into scope, but to generate the error only at the point that
the imported name was actually *used*. Then collisions could be resolved simply
by adding a single explicit, shadowing import.

**Default type parameters**

Some of the minor changes for moving to more generic code depends on an
interplay between defaulted type paramters and type inference, which has been
[accepted as an RFC](https://github.com/rust-lang/rfcs/pull/213) but not yet
implemented.

**"Extensible" enums**

There is already [an RFC](https://github.com/rust-lang/rfcs/pull/757) for an
`enum` annotation that would make it possible to add variants without ever
breaking downstream code.

**Sealed traits**

The ability to annotate a trait with some "sealed" marker, saying that no
external implementations are allowed, would be useful in certain cases where a
crate wishes to define a closed set of types that implements a particular
interface. Such an attribute would make it possible to evolve the interface
without a major version bump (since no downstream implementors can exist).

**Defaulted parameters**

Also known as "optional arguments" -- an
[oft-requested](https://github.com/rust-lang/rfcs/issues/323) feature. Allowing
arguments to a function to be optional makes it possible to add new arguments
after the fact without a major version bump.

**Open-ended explicit type paramters**

One hazard is that with today's explicit type parameter syntax, you must always
specify *all* type parameters: `foo::<T, U>(x, y)`. That means that adding a new
type parameter to `foo` can break code, even if a default is provided.

This could be easily addressed by adding a notation like `...` to leave
additional parameters unspecified: `foo::<T, ...>(x, y)`.

# Drawbacks and Alternatives

The main drawback to the approach laid out here is that it makes the stability
and semver guarantees a bit fuzzier: the promise is not that code will never
break, full stop, but rather that minor release breakage is of an extremely
limited form, for which there are a variety of mitigation strategies. This
approach tries to strike a middle ground between a very hard line for stability
(which, for Rust, would rule out many forms of extension) and willy-nilly
breakage: it's an explicit, but pragmatic policy.

An alternative would be to take a harder line and find some other way to allow
API evolution. Supposing that we resolved the issues around glob imports, the
main problems with breakage have to do with adding new inherent methods or trait
implementations -- both of which are vital forms of evolution. It might be
possible, in the standard library case, to provide some kind of version-based
opt in to this evolution: a crate could opt in to breaking changes for a
particular version of Rust, which might in turn be provided only through some
`cfg`-like mechanism.

Note that these strategies are not mutually exclusive. Rust's development
processes involved a very steady, strong stream of breakage, and while we need
to be very serious about stabilization, it is possible to take an iterative
approach. The changes considered "major" by this RFC already move the bar *very
significantly* from what was permitted pre-1.0. It may turn out that even the
minor forms of breakage permitted here are, in the long run, too much to
tolerate; at that point we could revise the policies here and explore some
opt-in scheme, for example.

# Unresolved questions

## Behavioral issues

- Is it permitted to change a contract from "abort" to "panic"? What about from
  "panic" to "return an `Err`"?

- Should we try to lay out more specific guidance for behavioral changes at this
  point?

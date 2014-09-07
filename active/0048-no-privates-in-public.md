- Start Date: 2014-06-24
- RFC PR #: [#136](https://github.com/rust-lang/rfcs/pull/136)
- Rust Issue #: [#16463](https://github.com/rust-lang/rust/issues/16463)

# Summary

Require a feature gate to expose private items in public APIs, until we grow the
appropriate language features to be able to remove the feature gate and forbid
it entirely.


# Motivation

Only those language features should be a part of 1.0 which we've intentionally
committed to supporting, and to preserving their behavior indefinitely.

The ability to mention private types in public APIs, and the apparent usefulness
of this ability for various purposes, is something that happened and was
discovered by accident. It also gives rise to various bizarre
[questions][questions] and interactions which we would need to expend effort
thinking about and answering, and which it would be much preferable to avoid
having to do entirely.

The only intentionally designed mechanism for representation hiding in the
current language is private fields in `struct`s (with newtypes as a specific
case), and this is the mechanism which should be used for it.

[questions]: https://github.com/rust-lang/rust/issues/10573


## Examples of strangeness

(See also https://github.com/rust-lang/rust/issues/10573.)

 * Being able to use a given type but not *write* the type is disconcerting.
   For example:

        struct Foo { ... }
        pub fn foo() -> Foo { ... }

   As a client of this module, I can write `let my_foo = foo();`, but not
   `let my_foo: Foo = foo();`. This is a logical consequence of the rules, but
   it doesn't make any sense.

 * Can I access public fields of a private type? For instance:

        struct Foo { pub x: int, ... }
        pub fn foo() -> Foo { ... }

   Can I now write `foo().x`, even though the struct `Foo` itself is not visible
   to me, and in `rust-doc`, I couldn't see its documentation? In other words,
   when the only way I could learn about the field `x` is by looking at the
   source code for the given module?

 * Can I call public methods on a private type?

 * Can I "know about" `trait`s the private type `impl`s?

Again, it's surely possible to formulate rules such that answers to these
questions can be deduced from them mechanically. But that doesn't mean it's a
good idea to do so. If the results are bizarre, then our assumptions should be
reconsidered. In these cases, it would be wiser to simply say, "don't do that".


## Properties

By restricting public APIs to only mentioning public items, we can guarantee that:

*Only public definitions are reachable through the public surface area of an API.*

Or in other words: for any item I see mentioned in `rust-doc` generated
documentation, I can *always* click it to see *its* documentation, in turn.

Or, dually:

*The presence or absence of private definitions should not be observable or
discoverable through the public API.*

As @aturon put it:

> One concrete problem with allowing private items to leak is that you lose some
> local reasoning. You might expect that if an item is marked private, you can
> refactor at will without breaking clients. But with leakage, you can't make
> this determination based on the item alone: you have to look at the entire API
> to spot leakages (or, I guess, have the lint do so for you). Perhaps not a
> huge deal in practice, but worrying nonetheless.


## Use cases for exposing private items, and preferable solutions

### Abstract types

One may wish to use a private type in a public API to hide its implementation,
either by using the private type in the API directly, or by defining a
`pub type`  synonym for it.

The correct solution in this case is to use a newtype instead. However, this can
currently be an unacceptably heavyweight solution in some cases, because one
must manually write all of the trait `impl`s to forward from the newtype to the
old type. This should be resolved by adding a [newtype deriving feature][gntd]
along the same lines as GHC (based on the same infrastructure as
[`Transmute`][91], née `Coercible`), or possibly with first-class module-scoped
existential types a la ML.

[gntd]: https://www.haskell.org/ghc/docs/7.8.1/html/users_guide/deriving.html#newtype-deriving
[91]: https://github.com/rust-lang/rfcs/pull/91


### Private supertraits

A use case for private supertraits currently is to prevent outside modules from
implementing the given trait, and potentially to have a private interface for
the given types which is accessible only from within the given module. For
example:

    trait PrivateInterface {
        fn internal_id(&self) -> uint;
    }

    pub trait PublicInterface: PrivateInterface {
        fn name(&self) -> String;
        ...
    }

    pub struct Foo { ... }
    pub struct Bar { ... }

    impl PrivateInterface for Foo { ... }
    impl PublicInterface  for Foo { ... }
    impl PrivateInterface for Bar { ... }
    impl PublicInterface  for Bar { ... }

    pub fn do_thing_with<T: PublicInterface>(x: &T) {
        // PublicInterface implies PrivateInterface!
        let id = x.internal_id();
        ...
    }

Here `PublicInterface` may only be implemented by us, because it requires
`PrivateInterface` as a supertrait, which is not exported outside the module.
Thus `PublicInterface` is only implemented by a closed set of types which we
specify. Public functions may require `PublicInterface` to be generic over this
closed set of types, and in their implementations, they may also use the methods
of the private `PrivateInterface` supertrait.

The better solution for this use case, which doesn't require exposing
a `PrivateInterface` in the public-facing parts of the API, would be to have
private trait methods. This can be seen by considering the analogy of `trait`s
as generic `struct`s and `impl`s as `static` instances of those `struct`s (with
the compiler selecting the appropriate instance based on type inference).
Supertraits can also be modelled as additional fields.

For example:

    pub trait Eq {
        fn eq(&self, other: &Self) -> bool;
        fn ne(&self, other: &Self) -> bool;
    }

    impl Eq for Foo {
        fn eq(&self, other: &Foo) -> bool { /* def eq */ }
        fn ne(&self, other: &Foo) -> bool { /* def ne */ }
    }

This corresponds to:

    pub struct Eq<Self> {
        pub eq: fn(&Self, &Self) -> bool,
        pub ne: fn(&Self, &Self) -> bool
    }

    pub static EQ_FOR_FOO: Eq<Foo> = {
        eq: |&this, &other| { /* def eq */ },
        ne: |&this, &other| { /* def ne */ }
    };

Now if we consider the private supertrait example from above, that becomes:

    struct PrivateInterface<Self> {
        pub internal_id: fn(&Self) -> uint
    }

    pub struct PublicInterface<Self> {
        pub super0: PrivateInterface<Self>,
        pub name: fn(&Self) -> String
    };

We can see that this solution is analogous to the same kind of
private-types-in-public-APIs situation which we want to forbid. And it sheds
light on a hairy question which had been laying hidden beneath the surface:
outside modules can't see `PrivateInterface`, but can they see `internal_id`?
We had been assuming "no", because that was convenient, but rigorously thinking
it through, `trait` methods are conceptually public, so this wouldn't
*necessarily* be the "correct" answer.

The *right* solution here is the same as for `struct`s: private fields, or
correspondingly, private methods. In other words, if we were working with
`struct`s and `static`s directly, we would write:

    pub struct PublicInterface<Self> {
        pub name: fn(&Self) -> String,
        internal_id: fn(&Self) -> uint
    }

so the public data is public and the private data is private, no mucking around
with the visibility of their *types*. Correspondingly, we would like to write
something like:

    pub trait PublicInterface {
        fn name(&self) -> String;
        priv fn internal_id(&self) -> uint;
    }

(Note that this is **not** a suggestion for particular syntax.)

If we can write this, everything we want falls out straightforwardly.
`internal_id` is only visible inside the given module, and outside modules can't
access it. Furthermore, just as you can't construct a (`static` or otherwise)
instance of a `struct` if it has inaccessible private fields, you also can't
construct an `impl` of a `trait` if it has inaccessible private methods.

So private supertraits should also be put behind a feature gate, like everything
else, until we figure out how to add private `trait` methods.


# Detailed design

## Overview

The general idea is that:

 * If an item is publicly exposed by a module `module`, items referred to in
   the public-facing parts of that item (e.g. its type) must themselves be
   public.

 * An item referred to in `module` is considered to be public if it is visible
   to clients of `module`.

Details follow.


## The rules

These rules apply as long as the feature gate is not enabled. After the feature
gate has been removed, they will apply always.

An item is considered to be publicly exposed by a module if it is declared `pub`
by that module, or if it is re-exported using `pub use` by that module.

For items which are publicly exposed by a module, the rules are that:

 * If it is a `static` declaration, items referred to in its type must be public.

 * If it is an `fn` declaration, items referred to in its trait bounds, argument
   types, and return type must be public.

 * If it is a `struct` or `enum` declaration, items referred to in its trait
   bounds and in the types of its `pub` fields must be public.

 * If it is a `type` declaration, items referred to in its definition must be
   public.

 * If it is a `trait` declaration, items referred to in its super-traits, in the
   trait bounds of its type parameters, and in the signatures of its methods
   (see `fn` case above) must be public.


## What does "public" mean?

An item `Item` referred to in the module `module` is considered to be public if:

 * The qualified name used by `module` to refer to `Item`, when recursively
   resolved through `use` declarations back to the original declaration of
   `Item`, resolves along the way to at least one `pub` declaration, whether a
   `pub use` declaration or a `pub` original declaration; and

 * For at least one of the above resolved-to `pub` declarations, all ancestor
   modules of the declaration, up to the deepest common ancestor module of the
   declaration with `module`, are `pub`.

In all other cases, an `Item` referred to in `module` is not considered to be
public, or `module` itself cannot refer to `Item` and the distinction is
irrelevant.

### Examples

In the following examples, the item `Item` referred to in the module `module`
is considered to be public:

````
pub mod module {
    pub struct Item { ... }
}
````

````
pub struct Item { ... }
pub mod module {
    use Item;
}
````

````
pub mod x {
    pub struct Item { ... }
}
pub mod module {
    use x::Item;
}
````

````
pub mod module {
    pub mod x {
        pub struct Item { ... }
    }
    use self::x::Item;
}
````

````
struct Item { ... }
pub mod module {
    pub use Item;
}
````

````
struct Foo { ... }
pub use Item = Foo;
pub mod module {
    use Item;
}
````

````
struct Foo { ... }
pub use Bar = Foo;
use Item = Bar;
pub mod module {
    use Item;
}
````

````
struct Item { ... }
pub mod x {
    pub use Item;
    pub mod y {
        use x::Item;
        pub mod module {
            use super::Item;
        }
    }
}
````

In the above examples, it is assumed that `module` will refer to `Item` as
simply `Item`, but the same thing holds true if `module` refrains from importing
`Item` explicitly with a private `use` declaration, and refers to it directly by
qualifying it with a path instead.


In the below examples, the item `Item` referred to in the module `module` is
*not* considered to be public:

````
pub mod module {
    struct Item { ... }
}
````

````
struct Item { ... }
pub mod module {
    use Item;
}
````

````
mod x {
    pub struct Item { ... }
}
pub mod module {
    use x::Item;
}
````

````
pub mod module {
    use self::x::Item;
    mod x {
        pub struct Item { ... }
    }
}
````

````
struct Item { ... }
pub use Alias = Item;
pub mod x {
    pub use Item;
    pub mod module {
        use Item; // refers to top-level `Item`
    }
}
````


# Drawbacks

Adds a (temporary) feature gate.

Requires some existing code to opt-in to the feature gate before transitioning
to saner alternatives.

Requires effort to implement.


# Alternatives

If we stick with the status quo, we'll have to resolve several bizarre questions
and keep supporting its behavior indefinitely after 1.0.

Instead of a feature gate, we could just ban these things outright right away,
at the cost of temporarily losing some convenience and a small amount of
expressiveness before the more principled replacement features are implemented.

We could make an exception for private supertraits, as these are not quite as
problematic as the other cases. However, especially given that a more principled
alternative is known (private methods), I would rather not make any exceptions.


# Unresolved questions

Is this the right set of rules to apply?

Did I describe them correctly in the "Detailed design"?

Did I miss anything? Are there any holes or contradictions?

Is there a simpler, easier, and/or more logical formulation of the rules?

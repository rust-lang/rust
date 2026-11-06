# Binder constraint testing DSL

As part of the [assumptions on binders](https://github.com/rust-lang/project-assumptions-on-binders) work (henceforth
just "abby"), a testing [DSL](https://en.wikipedia.org/wiki/Domain-specific_language) was implemented in rustc to make
tests for abby easier to write. Writing the equivalent surface-level rust is extremely verbose, difficult to produce,
very opaque to read, and rough to maintain.

This document describes that DSL.

# Basic syntax

The DSL is implemented via an internal macro, under the perma-unstable `#![feature(test_binder_constraints)]`

```rust
core::test_binder_constraints! {
    impl<'a, 'b> where 'a: 'b {
        'a: 'b
    }
}
```

The top-level syntax is the keyword `impl`, followed by an option list of generic args, followed by an optional `where`
clause. This is extremely similar to an `impl` block (hence the keyword choice), a free `fn`, or whatever else top-level
kind of generic thing. This lowers to your typical typeck root DefId with `generics_of` being the written generics +
constraints. These generics are your typical `TyKind::Param`, and are *NOT* `ty::Binder`s.

# Registering constraints to be proven in a body

Inside the main block, there are two different syntaxes to register constraints. These go through two fairly different
pipelines.

### "Direct insert" syntax

The first syntax is one that directly inserts the written constraints into constraint storage:

```rust
core::test_binder_constraints! {
    impl<'a, 'b, T> {
        'a: 'b, // produces a LeafRegionConstraint::RegionOutlives
        T: 'b, // produces a LeafRegionConstraint::PlaceholderTyOutlives
        for<'c> T::Assoc<'c>: 'b, // produces a LeafRegionConstraint::AliasTyOutlivesViaEnv
        ambiguity, // produces a LeafRegionConstraint::Ambiguity
    }
}
```

Note that PlaceholderTyOutlives and AliasTyOutlivesViaEnv are vaguely ambiguous what you "mean" when you just write a
type outlives, so: a plain `SomeType: 'lifetime` always produces a PlaceholderTyOutlives, and `for<..> SomeType:
'lifetime` always produces a AliasTyOutlivesViaEnv. If you want an AliasTyOutlivesViaEnv without bound variables, just
write an empty `for<>`, like `for<> T::Assoc: 'b`. PlaceholderTyOutlives must be a placeholder (i.e. bound variable
instantiated with a placeholder when entering a binder, or top-level param) on the LHS, and AliasTyOutlivesViaEnv must
be an alias. The test infrastructure will yell at you and error otherwise.

Additionally, "direct insert" style constraints support `and { }` and `or { }` syntax, to combine constraints together
(the top-level body can be thought of being implicitly wrapped in an `and { }`):

```rust
core::test_binder_constraints! {
    impl<'a, 'b, 'c> {
        or {
            and {
                'a: 'b,
                'a: 'c,
            },
            'b: 'c,
            'b: 'a,
        }
    }
}
```

This example corresponds to, conceptually, something like `(('a: 'b) && ('a: 'c)) || ('b: 'c) || ('b: 'a)`. These things
can be arbitrarily nested (they are converted to disjunctive normal form / "canonical form" automatically).

"Direct insert" constraints are parsed with a custom DSL parser, so fancy syntax like `'a: 'b + 'c` is not supported.
Just write multiple constraints instead.

### `predicates` syntax

The other main syntax is with the `predicates` keyword:

```rust
core::test_binder_constraints! {
    impl<'a, 'b, 'c, T: Trait> {
        predicates T::Assoc: 'a + 'b, T::Assoc: 'c
    }
}
```

These are parsed identically to a `where` clause, and so support all sorts of fancy Rust syntax, like `'a: 'b + 'c`.
Rustc obligations are created from these clauses and registered as things to prove via `register_obligation` (and as
such are later evaluated resulting in a pile of region constraints and
[whatnot](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/traits/solve/struct.ExternalConstraintsData.html)).
The LHS is not restricted to just being a placeholder or alias. These are much more "normal" Rust syntax, and also makes
it possible to test the `register_obligation` pipeline rather than just registering constraints directly into the
context.

`and { }` and `or { }` syntax is NOT supported with the `predicates` syntax (because `register_obligation` is oblivious
to `or` semantics, you effectively can only register `and` stuff)

# Existential quantification

Writing `exists`, followed by a generic args list, introduces new infer vars. `where` clauses on this generics args list
don't really make sense and are not a thing. The body of the `exists` supports everything the body of the top-level
`impl` is, it's just another body.

Only lifetimes are supported here, not types, unless you enable `#![feature(non_lifetime_binders)]`.

```rust
core::test_binder_constraints! {
    impl<'a> {
        exists<'b> {
            'a: 'b,
            'b: 'a,
        }
    }
}
```

# Universal quantification

This is the main point of the testing DSL. ([universal
quantifiers](https://en.wikipedia.org/wiki/Universal_quantification) are also called "for all"s in type theory, hence
the rust syntax `for<'a>`, and the test DSL syntax `forall<'a>`, and in rustc are called `ty::Binder`)

```rust
core::test_binder_constraints! {
    impl<'b, 'c: 'b> {
        forall<'a> where 'b: 'a {
            'c: 'a
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}
```

There's several components to this:

`forall<'a>` is a generic parameter list for the binder - only lifetimes are supported here, unless you enable
`#![feature(non_lifetime_binders)]`.

`where 'b: 'a` are the *assumptions* on the binder. The where clauses, the conditions that must be valid for the binder
to be instantiated, many names for it. You *must* write out all elaborations of any where clauses *explicitly*. If you
forget to do so, the test infrastructure will yell and throw an error with the missing clause at you.

Then, the body: `{ 'c: 'a }`. This supports everything the body of the top-level `impl` supports: direct-insert
constraints (including `and`/`or`), `predicates` constraints, additional `exists` and `forall` clauses

Finally, there are a list of constraints that we test and assert are present upon exiting the `forall` binder. This
*only* supports the "direct insert constraint" style of syntax, it does *not* support `predicates` syntax. It must
exactly match each and every constraint present when exiting the binder.

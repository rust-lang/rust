# Lowering rules

This section gives the complete lowering rules for Rust traits into
[program clauses][pc]. It is a kind of reference. These rules
reference the [domain goals][dg] defined in an earlier section.

[pc]: ./goals-and-clauses.html
[dg]: ./goals-and-clauses.html#domain-goals

## Notation

The nonterminal `Pi` is used to mean some generic *parameter*, either a
named lifetime like `'a` or a type parameter like `A`.

The nonterminal `Ai` is used to mean some generic *argument*, which
might be a lifetime like `'a` or a type like `Vec<A>`.

When defining the lowering rules, we will give goals and clauses in
the [notation given in this section](./goals-and-clauses.html).
We sometimes insert "macros" like `LowerWhereClause!` into these
definitions; these macros reference other sections within this chapter.

## Rule names and cross-references

Each of these lowering rules is given a name, documented with a
comment like so:

    // Rule Foo-Bar-Baz

The reference implementation of these rules is to be found in
[`chalk/chalk-rules/src/clauses.rs`][chalk_rules]. They are also ported in
rustc in the [`librustc_traits`][librustc_traits] crate.

[chalk_rules]: https://github.com/rust-lang/chalk/blob/master/chalk-rules/src/clauses.rs
[librustc_traits]: https://github.com/rust-lang/rust/tree/master/src/librustc_traits

## Lowering where clauses

When used in a goal position, where clauses can be mapped directly to
the `Holds` variant of [domain goals][dg], as follows:

- `A0: Foo<A1..An>` maps to `Implemented(A0: Foo<A1..An>)`
- `T: 'r` maps to `Outlives(T, 'r)`
- `'a: 'b` maps to `Outlives('a, 'b)`
- `A0: Foo<A1..An, Item = T>` is a bit special and expands to two distinct
  goals, namely `Implemented(A0: Foo<A1..An>)` and
  `ProjectionEq(<A0 as Foo<A1..An>>::Item = T)`

In the rules below, we will use `WC` to indicate where clauses that
appear in Rust syntax; we will then use the same `WC` to indicate
where those where clauses appear as goals in the program clauses that
we are producing. In that case, the mapping above is used to convert
from the Rust syntax into goals.

### Transforming the lowered where clauses

In addition, in the rules below, we sometimes do some transformations
on the lowered where clauses, as defined here:

- `FromEnv(WC)` – this indicates that:
  - `Implemented(TraitRef)` becomes `FromEnv(TraitRef)`
  - other where-clauses are left intact
- `WellFormed(WC)` – this indicates that:
  - `Implemented(TraitRef)` becomes `WellFormed(TraitRef)`
  - other where-clauses are left intact

*TODO*: I suspect that we want to alter the outlives relations too,
but Chalk isn't modeling those right now.

## Lowering traits

Given a trait definition

```rust,ignore
trait Trait<P1..Pn> // P0 == Self
where WC
{
    // trait items
}
```

we will produce a number of declarations. This section is focused on
the program clauses for the trait header (i.e., the stuff outside the
`{}`); the [section on trait items](#trait-items) covers the stuff
inside the `{}`.

### Trait header

From the trait itself we mostly make "meta" rules that setup the
relationships between different kinds of domain goals.  The first such
rule from the trait header creates the mapping between the `FromEnv`
and `Implemented` predicates:

```text
// Rule Implemented-From-Env
forall<Self, P1..Pn> {
  Implemented(Self: Trait<P1..Pn>) :- FromEnv(Self: Trait<P1..Pn>)
}
```

<a name="implied-bounds"></a>

#### Implied bounds

The next few clauses have to do with implied bounds (see also
[RFC 2089] and the [implied bounds][implied_bounds] chapter for a more in depth
cover). For each trait, we produce two clauses:

[RFC 2089]: https://rust-lang.github.io/rfcs/2089-implied-bounds.html
[implied_bounds]: ./implied-bounds.md

```text
// Rule Implied-Bound-From-Trait
//
// For each where clause WC:
forall<Self, P1..Pn> {
  FromEnv(WC) :- FromEnv(Self: Trait<P1..Pn)
}
```

This clause says that if we are assuming that the trait holds, then we can also
assume that its where-clauses hold. It's perhaps useful to see an example:

```rust,ignore
trait Eq: PartialEq { ... }
```

In this case, the `PartialEq` supertrait is equivalent to a `where
Self: PartialEq` where clause, in our simplified model. The program
clause above therefore states that if we can prove `FromEnv(T: Eq)` –
e.g., if we are in some function with `T: Eq` in its where clauses –
then we also know that `FromEnv(T: PartialEq)`. Thus the set of things
that follow from the environment are not only the **direct where
clauses** but also things that follow from them.

The next rule is related; it defines what it means for a trait reference
to be **well-formed**:

```text
// Rule WellFormed-TraitRef
forall<Self, P1..Pn> {
  WellFormed(Self: Trait<P1..Pn>) :- Implemented(Self: Trait<P1..Pn>) && WellFormed(WC)
}
```

This `WellFormed` rule states that `T: Trait` is well-formed if (a)
`T: Trait` is implemented and (b) all the where-clauses declared on
`Trait` are well-formed (and hence they are implemented). Remember
that the `WellFormed` predicate is
[coinductive](./goals-and-clauses.html#coinductive); in this
case, it is serving as a kind of "carrier" that allows us to enumerate
all the where clauses that are transitively implied by `T: Trait`.

An example:

```rust,ignore
trait Foo: A + Bar { }
trait Bar: B + Foo { }
trait A { }
trait B { }
```

Here, the transitive set of implications for `T: Foo` are `T: A`, `T: Bar`, and
`T: B`.  And indeed if we were to try to prove `WellFormed(T: Foo)`, we would
have to prove each one of those:

- `WellFormed(T: Foo)`
  - `Implemented(T: Foo)`
  - `WellFormed(T: A)`
    - `Implemented(T: A)`
  - `WellFormed(T: Bar)`
    - `Implemented(T: Bar)`
    - `WellFormed(T: B)`
      - `Implemented(T: Bar)`
    - `WellFormed(T: Foo)` -- cycle, true coinductively

This `WellFormed` predicate is only used when proving that impls are
well-formed – basically, for each impl of some trait ref `TraitRef`,
we must show that `WellFormed(TraitRef)`. This in turn justifies the
implied bounds rules that allow us to extend the set of `FromEnv`
items.

## Lowering type definitions

We also want to have some rules which define when a type is well-formed.
For example, given this type:

```rust,ignore
struct Set<K> where K: Hash { ... }
```

then `Set<i32>` is well-formed because `i32` implements `Hash`, but
`Set<NotHash>` would not be well-formed. Basically, a type is well-formed
if its parameters verify the where clauses written on the type definition.

Hence, for every type definition:

```rust, ignore
struct Type<P1..Pn> where WC { ... }
```

we produce the following rule:

```text
// Rule WellFormed-Type
forall<P1..Pn> {
  WellFormed(Type<P1..Pn>) :- WC
}
```

Note that we use `struct` for defining a type, but this should be understood
as a general type definition (it could be e.g. a generic `enum`).

Conversely, we define rules which say that if we assume that a type is
well-formed, we can also assume that its where clauses hold. That is,
we produce the following family of rules:

```text
// Rule Implied-Bound-From-Type
//
// For each where clause `WC`
forall<P1..Pn> {
  FromEnv(WC) :- FromEnv(Type<P1..Pn>)
}
```

As for the implied bounds RFC, functions will *assume* that their arguments
are well-formed. For example, suppose we have the following bit of code:

```rust,ignore
trait Hash: Eq { }
struct Set<K: Hash> { ... }

fn foo<K>(collection: Set<K>, x: K, y: K) {
    // `x` and `y` can be equalized even if we did not explicitly write
    // `where K: Eq`
    if x == y {
        ...
    }
}
```

In the `foo` function, we assume that `Set<K>` is well-formed, i.e. we have
`FromEnv(Set<K>)` in our environment. Because of the previous rule, we get
 `FromEnv(K: Hash)` without needing an explicit where clause. And because
of the `Hash` trait definition, there also exists a rule which says:

```text
forall<K> {
  FromEnv(K: Eq) :- FromEnv(K: Hash)
}
```

which means that we finally get `FromEnv(K: Eq)` and then can compare `x`
and `y` without needing an explicit where clause.

<a name="trait-items"></a>

## Lowering trait items

### Associated type declarations

Given a trait that declares a (possibly generic) associated type:

```rust,ignore
trait Trait<P1..Pn> // P0 == Self
where WC
{
    type AssocType<Pn+1..Pm>: Bounds where WC1;
}
```

We will produce a number of program clauses. The first two define
the rules by which `ProjectionEq` can succeed; these two clauses are discussed
in detail in the [section on associated types](./associated-types.html),
but reproduced here for reference:

```text
// Rule ProjectionEq-Normalize
//
// ProjectionEq can succeed by normalizing:
forall<Self, P1..Pn, Pn+1..Pm, U> {
  ProjectionEq(<Self as Trait<P1..Pn>>::AssocType<Pn+1..Pm> = U) :-
      Normalize(<Self as Trait<P1..Pn>>::AssocType<Pn+1..Pm> -> U)
}
```

```text
// Rule ProjectionEq-Placeholder
//
// ProjectionEq can succeed through the placeholder associated type,
// see "associated type" chapter for more:
forall<Self, P1..Pn, Pn+1..Pm> {
  ProjectionEq(
    <Self as Trait<P1..Pn>>::AssocType<Pn+1..Pm> =
    (Trait::AssocType)<Self, P1..Pn, Pn+1..Pm>
  )
}
```

The next rule covers implied bounds for the projection. In particular,
the `Bounds` declared on the associated type must have been proven to hold
to show that the impl is well-formed, and hence we can rely on them
elsewhere.

```text
// Rule Implied-Bound-From-AssocTy
//
// For each `Bound` in `Bounds`:
forall<Self, P1..Pn, Pn+1..Pm> {
    FromEnv(<Self as Trait<P1..Pn>>::AssocType<Pn+1..Pm>>: Bound) :-
      FromEnv(Self: Trait<P1..Pn>) && WC1
}
```

Next, we define the requirements for an instantiation of our associated
type to be well-formed...

```text
// Rule WellFormed-AssocTy
forall<Self, P1..Pn, Pn+1..Pm> {
    WellFormed((Trait::AssocType)<Self, P1..Pn, Pn+1..Pm>) :-
      Implemented(Self: Trait<P1..Pn>) && WC1
}
```

...along with the reverse implications, when we can assume that it is
well-formed.

```text
// Rule Implied-WC-From-AssocTy
//
// For each where clause WC1:
forall<Self, P1..Pn, Pn+1..Pm> {
    FromEnv(WC1) :- FromEnv((Trait::AssocType)<Self, P1..Pn, Pn+1..Pm>)
}
```

```text
// Rule Implied-Trait-From-AssocTy
forall<Self, P1..Pn, Pn+1..Pm> {
    FromEnv(Self: Trait<P1..Pn>) :-
      FromEnv((Trait::AssocType)<Self, P1..Pn, Pn+1..Pm>)
}
```

### Lowering function and constant declarations

Chalk didn't model functions and constants, but I would eventually like to
treat them exactly like normalization. See [the section on function/constant
values below](#constant-vals) for more details.

## Lowering impls

Given an impl of a trait:

```rust,ignore
impl<P0..Pn> Trait<A1..An> for A0
where WC
{
    // zero or more impl items
}
```

Let `TraitRef` be the trait reference `A0: Trait<A1..An>`. Then we
will create the following rules:

```text
// Rule Implemented-From-Impl
forall<P0..Pn> {
  Implemented(TraitRef) :- WC
}
```

In addition, we will lower all of the *impl items*.

## Lowering impl items

### Associated type values

Given an impl that contains:

```rust,ignore
impl<P0..Pn> Trait<P1..Pn> for P0
where WC_impl
{
    type AssocType<Pn+1..Pm> = T;
}
```

and our where clause `WC1` on the trait associated type from above, we
produce the following rule:

```text
// Rule Normalize-From-Impl
forall<P0..Pm> {
  forall<Pn+1..Pm> {
    Normalize(<P0 as Trait<P1..Pn>>::AssocType<Pn+1..Pm> -> T) :-
      Implemented(P0 as Trait) && WC1
  }
}
```

Note that `WC_impl` and `WC1` both encode where-clauses that the impl can
rely on. (`WC_impl` is not used here, because it is implied by
`Implemented(P0 as Trait)`.)

<a name="constant-vals"></a>

### Function and constant values

Chalk didn't model functions and constants, but I would eventually
like to treat them exactly like normalization. This presumably
involves adding a new kind of parameter (constant), and then having a
`NormalizeValue` domain goal. This is *to be written* because the
details are a bit up in the air.

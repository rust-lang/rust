- Feature Name: associated_type_bounds
- Start Date: 2018-01-13
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Introduce the bound form `MyTrait<AssociatedType: Bounds>`, permitted anywhere
a bound of the form `MyTrait<AssociatedType = T>` would be allowed. The bound
`T: Trait<AssociatedType: Bounds>` desugars to the bounds `T: Trait` and
`<T as Trait>::AssociatedType: Bounds`.
See the [reference][reference-level-explanation] and [rationale][alternatives]
for exact details.

# Motivation
[motivation]: #motivation

Currently, when specifying a bound using a trait that has an associated
type, the developer can specify the precise type via the syntax
`MyTrait<AssociatedType = T>`. With the introduction of the `impl Trait`
syntax for static-dispatch existential types, this syntax also permits
`MyTrait<AssociatedType = impl Bounds>`, as a shorthand for introducing a
new type variable and specifying those bounds.

However, this introduces an unnecessary level of indirection that does not
match the developer's intuition and mental model as well as it could. In
particular, given the ability to write bounds on a type variable as `T: Bounds`,
it makes sense to permit writing bounds on an associated type directly.
This results in the simpler syntax `MyTrait<AssociatedType: Bounds>`.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Instead of specifying a concrete type for an associated type, we can
specify a bound on the associated type, to ensure that it implements
specific traits, as seen in the example below:

```rust
fn print_all<T: Iterator<Item: Display>>(printables: T) {
    for p in printables {
        println!("{}", p);
    }
}
```

## In anonymous existential types

```rust
fn printables() -> impl Iterator<Item: Display> {
    // ..
}
```

## Further examples

Instead of writing:

```rust
impl<I> Clone for Peekable<I>
where
    I: Clone + Iterator,
    <I as Iterator>::Item: Clone,
{
    // ..
}
```

you may write:

```rust
impl<I> Clone for Peekable<I>
where
    I: Clone + Iterator<Item: Clone>
{
    // ..
}
```

or replace the `where` clause entirely:

```rust
impl<I: Clone + Iterator<Item: Clone>> Clone for Peekable<I> {
    // ..
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The surface syntax `T: Trait<AssociatedType: Bounds>` should desugar to a pair
of bounds: `T: Trait` and `<T as Trait>::AssociatedType: Bounds`.
Rust currently allows both of those bounds anywhere a bound can currently appear;
the new syntax does not introduce any new semantics.

Additionally, the surface syntax `impl Trait<AssociatedType: Bounds>` turns
into a named type variable `T`, universal or existential depending on context,
with the usual bound `T: Trait` along with the added bound
`<T as Trait>::AssociatedType: Bounds`.

Meanwhile, the surface syntax `dyn Trait<AssociatedType: Bounds>` desugars into
`dyn Trait<AssociatedType = T>` where `T` is a named type variable `T` with the
bound `T: Bounds`.

## The desugaring for associated types

In the case of an associated type having a bound of the form:

```rust
trait TraitA {
    type AssocA: TraitB<AssocB: TraitC>;
}
```

we desugar to an anonymous associated type for `AssocB`, which corresponds to:

```rust
trait TraitA {
    type AssocA: TraitB<AssocB = Self::AssocA_0>;
    type AssocA_0: TraitC; // Associated type is Unnamed!
}
```

## Notes on the meaning of `impl Trait<Assoc: Bound>`

Note that in the context `-> impl Trait<Assoc: Bound>`, since the Trait is
existentially quantified, the `Assoc` is as well. Semantically speaking,
`fn printables..` is equivalent to:

```rust
fn printables() -> impl Iterator<Item = impl Display> { .. }
```

For `arg: impl Trait<Assoc: Bound>`, it can likewise be seen as:
`arg: impl Trait<Assoc = impl Bound>`.

## Meaning of `existential type Foo: Trait<Assoc: Bound>`

Given:

```
existential type Foo: Trait<Assoc: Bound>;
```

it can be seen as the same as:

```rust
existential type Foo: Trait<Assoc = _0>;
existential type _0: Bound;
```

[RFC 2071]: https://github.com/rust-lang/rfcs/blob/master/text/2071-impl-trait-type-alias.md

This syntax is specified in [RFC 2071]. As in that RFC, this documentation
uses the non-final syntax for existential type aliases.

# Drawbacks
[drawbacks]: #drawbacks

Rust code can already express this using the desugared form. This proposal
just introduces a simpler surface syntax that parallels other uses of bounds.
As always, when introducing new syntactic forms, an increased burden is put on
developers to know about and understand those forms, and this proposal is no
different. However, we believe that the parallel to the use of bounds elsewhere
makes this new syntax immediately recognizable and understandable.

# Rationale and alternatives
[alternatives]: #rationale-and-alternatives

As with any new surface syntax, one alternative is simply not introducing
the syntax at all. That would still leave developers with the
`MyTrait<AssociatedType = impl Bounds>` form. However, allowing the more
direct bounds syntax provides a better parallel to the use of bounds elsewhere.
The introduced form in this RFC is comparatively both shorter and clearer.

### An alternative desugaring of bounds on associated types

[RFC 2089]: https://github.com/rust-lang/rfcs/blob/master/text/2089-implied-bounds.md

An alternative desugaring of the following definition:

```rust
trait TraitA {
    type AssocA: TraitB<AssocB: TraitC>;
}
```

is to add the `where` clause, as specified above, to the trait, desugaring to:

```rust
trait TraitA
where
    <Self::AssocA as TraitB>::AssocB: TraitC,
{
    type AssocA: TraitB;
}
```

However, at the time of this writing, a Rust compiler will treat this
differently than the desugaring proposed in the reference.
The following snippet illustrates the difference:

```rust
trait Foo where <Self::Bar as Iterator>::Item: Copy {
    type Bar: Iterator;
}

trait Foo2 {
    type Bar: Iterator<Item = Self::BarItem>;
    type BarItem: Copy;
}

fn use_foo<X: Foo>(arg: X)
where <X::Bar as Iterator>::Item: Copy
// ^-- Remove this line and it will error with:
// error[E0277]: `<<X as Foo>::Bar as std::iter::Iterator>::Item` doesn't implement `Copy`
{
    let item: <X::Bar as Iterator>::Item;
}

fn use_foo2<X: Foo2>(arg: X) {
    let item: <X::Bar as Iterator>::Item;
}
```

The desugaring with a `where` therefore becomes problematic from a perspective
of usability.

However, [RFC 2089, Implied Bounds][RFC 2089] specifies that desugaring to the
`where` clause in the trait will permit the `use_foo` function to omit its
`where` clause. This entails that both desugarings become equivalent from the
point of view of a user. The desugaring with `where` therefore becomes viable
in the presence of [RFC 2089].

# Unresolved questions
[unresolved]: #unresolved-questions

- Does allowing this for `dyn` trait objects introduce any unforeseen issues?
  This can be resolved during stabilization.

- The exact desugaring in the context of putting bounds on an associated type
  of a trait is left unresolved. The semantics should however be preserved.
  This is also the case with other desugarings in this RFC.

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

The surface syntax `T: Trait<AssociatedType: Bounds>` should always desugar
to a pair of bounds: `T: Trait` and `<T as Trait>::AssociatedType: Bounds`.
Rust currently allows both of those bounds anywhere a bound can currently appear;
the new syntax does not introduce any new semantics.

Additionally, the surface syntax `impl Trait<AssociatedType: Bounds>` turns
into a named type variable `T`, universal or existential depending on context,
with the usual bound `T: Trait` along with the added bound
`<T as Trait>::AssociatedType: Bounds`.

Meanwhile, the surface syntax `dyn Trait<AssociatedType: Bounds>` desugars into
`dyn Trait<AssociatedType = T>` where `T` is a named type variable `T` with the
bound `T: Bounds`.

# Drawbacks
[drawbacks]: #drawbacks

Rust code can already express this using the desugared form. This proposal
just introduces a simpler surface syntax that parallels other uses of bounds.
As always, when introducing new syntactic forms, an increased burden is put on
developers to know about and understand those forms, and this proposal is no
different. However, we believe that the parallel to the use of bounds elsewhere
makes this new syntax immediately recognizable and understandable.

# Rationale and alternatives
[alternatives]: #alternatives

As with any new surface syntax, one alternative is simply not introducing
the syntax at all. That would still leave developers with the
`MyTrait<AssociatedType = impl Bounds>` form. However, allowing the more
direct bounds syntax provides a better parallel to the use of bounds elsewhere.
The introduced form in this RFC is comparatively both shorter and clearer.

# Unresolved questions
[unresolved]: #unresolved-questions

- Does allowing this for `dyn` trait objects introduce any unforseen issues?
  This can be resolved during stabilization.
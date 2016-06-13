- Feature Name: `allow_self_in_where_clauses`
- Start Date: 2016-06-13
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC proposes allowing the `Self` type to be used in where clauses for trait
implementations, as well as referencing associated types for the trait being
implemented.

# Motivation
[motivation]: #motivation

`Self` is a useful tool to have to reduce churn when the type changes for
various reasons. One would expect to be able to write

```rust
impl SomeTrait for MySuperLongType<T, U, V, W, X> where
  Self: SomeOtherTrait,
```

but this will fail to compile today, forcing you to repeat the type, and adding
one more place that has to change if the type ever changes.

By this same logic, we would also like to be able to reference associated types
from the traits being implemented. When dealing with generic code, patterns like
this often emerge:

```rust
trait MyTrait {
    type MyType: SomeBound;
}

impl<T, U, V> MyTrait for SomeStruct<T, U, V> where
    SomeOtherStruct<T, U, V>: SomeBound,
{
    type MyType = SomeOtherStruct<T, U, V>;
}
```

the only reason the associated type is repeated at all is to restate the bound
on the associated type. It would be nice to reduce some of that duplication.

# Detailed design
[design]: #detailed-design

The first half of this RFC is simple. Inside of a where clause for trait
implementations, `Self` will refer to the type the trait is being implemented
for. It will have the same value as `Self` being used in the body of the trait
implementation.

Accessing associated types will have the same result as copying the body of the
associated type into the place where it's being used. That is to say that it
will assume that all constraints hold, and evaluate to what the type would have
been in that case. Ideally one should never have to write `<Self as
CurrentTrait>::SomeType`, but in practice it will likely be required to remove
issues with recursive evaluation.

# Drawbacks
[drawbacks]: #drawbacks

`Self` is always less explicit than the alternative

# Alternatives
[alternatives]: #alternatives

Not implementing this, or only allowing bare `Self` but not associated types in
where clauses

# Unresolved questions
[unresolved]: #unresolved-questions

None

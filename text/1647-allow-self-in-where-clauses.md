- Feature Name: `allow_self_in_where_clauses`
- Start Date: 2016-06-13
- RFC PR: [#1647](https://github.com/rust-lang/rfcs/pull/1647)
- Rust Issue: [#38864](https://github.com/rust-lang/rust/issues/38864)

# Summary
[summary]: #summary

This RFC proposes allowing the `Self` type to be used in every position in trait
implementations, including where clauses and other parameters to the trait being
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

Instead of blocking `Self` from being used in the "header" of a trait impl,
it will be understood to be a reference to the implementation type. For example,
all of these would be valid:

```rust
impl SomeTrait for SomeType where Self: SomeOtherTrait { }

impl SomeTrait<Self> for SomeType { }

impl SomeTrait for SomeType where SomeOtherType<Self>: SomeTrait { }

impl SomeTrait for SomeType where Self::AssocType: SomeOtherTrait {
    AssocType = SomeOtherType;
}
```

If the `Self` type is parameterized by `Self`, an error that the type definition
is recursive is thrown, rather than not recognizing self.

```rust
// The error here is because this would be Vec<Vec<Self>>, Vec<Vec<Vec<Self>>>, ...
impl SomeTrait for Vec<Self> { }
```

# Drawbacks
[drawbacks]: #drawbacks

`Self` is always less explicit than the alternative.

# Alternatives
[alternatives]: #alternatives

Not implementing this is an alternative, as is accepting Self only in where clauses
and not other positions in the impl header.

# Unresolved questions
[unresolved]: #unresolved-questions

None

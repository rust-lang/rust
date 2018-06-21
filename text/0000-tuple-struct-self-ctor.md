- Feature Name: tuple_struct_self_ctor
- Start Date: 2017-01-18
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Tuple `struct`s can now be constructed and pattern matched with
`Self(v1, v2, ..)`. A simple example:

```rust
struct TheAnswer(usize);

impl Default for TheAnswer {
    fn default() -> Self { Self(42) }
}
```

Similarly, unit structs can also be constructed and pattern matched with `Self`.

# Motivation
[motivation]: #motivation

This RFC proposes a consistency fix allowing `Self` to be used in more
places to better match the users' intuition of the language and to get
closer to feature parity between tuple structs and structs with named fields.

Currently, only structs with named fields can be constructed inside
impls using `Self` like so:

```rust
struct Mascot { name: String, age: usize }

impl Default for Mascot {
    fn default() -> Self {
        Self {
            name: "Ferris the Crab".into(),
            age: 3
        }
    }
}
```

while the following is not allowed:

```rust
struct Mascot(String, usize);

impl Default for Mascot {
    fn default() -> Self {
        Self("Ferris the Crab".into(), 3)
    }
}
```

This discrepancy is unfortunate as many users reach for `Self(v0, v1, ..)`
from time to time, only to find that it doesn't work. This creates a break
in the users intuition and becomes a papercut. It also has the effect that
each user must remember this exception, making the rule-set to remember
larger wherefore the language becomes more complex.

There are good reasons why `Self { f0: v0, f1: v1, .. }` is allowed.
Chiefly among those is that it becomes easier to refactor the code when
one wants to rename type names. Another important reason is that only
having to keep `Self` in mind means that a developer does not need to
keep the type name fresh in their working memory. This is beneficial for
users with shorter working memory such as the author of this RFC.

Since `Self { f0: v0, .. }` is well motivated, those benefits and motivations
will also extend to tuple and unit structs. Eliminating this discrepancy between
tuple structs and those with named fields will therefore have all the benefits
associated with this feature for structs with named fields.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Basic concept

For structs with named fields such as:

```rust
struct Person {
    name: String,
    ssn: usize,
    age: usize
}
```

You may use the syntax `Self { field0: value0, .. }` as seen below
instead of writing `TypeName { field0: value0, .. }`:

```rust
impl Person {
    /// Make a newborn person.
    fn newborn(name: String, ssn: usize) -> Self {
        Self { name, ssn, age: 0 }
    }
}
```

## Through type aliases

This ability does not extend to tuple structs however in current Rust but will
with this RFC. To continue on with the previous example, you can now also write:

```rust
struct Person(String, usize, usize);

impl Person {
    /// Make a newborn person.
    fn newborn(name: String, ssn: usize) -> Self {
        Self(name, ssn, 0)
    }
}
```

As with structs with named fields, you may also use `Self` when
you are `impl`ing on a type alias of a struct as seen here:

```rust
struct FooBar(u8);

type BarFoo = FooBar;

impl Default for BarFoo {
    fn default() -> Self {
        Self(42) // <-- Not allowed before.
    }
}
```

## Patterns

Currently, you can pattern match using `Self { .. }` on a named struct as in
the following example:

```rust
struct Person {
    ssn: usize,
    age: usize
}

impl Person {
    /// Make a newborn person.
    fn newborn(ssn: usize) -> Self {
        match { Self { ssn, age: 0 } } {
            Self { ssn, age } // `Self { .. }` is permitted as a pattern!
                => Self { ssn, age }
        }
    }
}
```

This RFC extends this to tuple structs:

```rust
struct Person(usize, usize);

impl Person {
    /// Make a newborn person.
    fn newborn(ssn: usize) -> Self {
        match { Self(ssn, 0) } {
            Self(ssn, age) // `Self(..)` is permitted as a pattern!
                => Self(ssn, age)
        }
    }
}
```

Of course, this redundant reconstruction is not recommended in actual code,
but illustrates what you can do.

## `Self` as a function pointer

When you define a tuple struct today such as:

```rust
struct Foo<T>(T);

impl<T> Foo<T> {
    fn fooify_iter(iter: impl Iterator<Item = T>) -> impl Iterator<Item = Foo<T>> {
        iter.map(Foo)
    }
}
```

you can use `Foo` as a function pointer typed at: `for<T> fn(T) -> T` as
seen in the example above.

This RFC extends that such that `Self` can also be used as a function pointer
for tuple structs. Modifying the example above gives us:

```rust
impl<T> Foo<T> {
    fn fooify_iter(iter: impl Iterator<Item = T>) -> impl Iterator<Item = Foo<T>> {
        iter.map(Self)
    }
}
```

## Unit structs

With this RFC, you can also use `Self` in pattern and expression contexts when
dealing with unit structs. For example:

```rust
struct TheAnswer;

impl Default for TheAnswer {
    fn default() -> Self {
        match { Self } { Self => Self }
    }
}
```

## Teaching the contents

This RFC should not require additional effort other than spreading the
news that this now is possible as well as the reference. The changes are
seen as intuitive enough that it supports what the user already assumes
should work and will probably try at some point.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Grammar

Given:

```
%token SELF
```

which lexes `Self`, the following are legal productions in the language:

```
pat : ... // <-- The original grammar of `pat` prior to this RFC.
    | SELF '(' ')'
    | SELF '(' pat_tup ')'
    | SELF
    | ...
    ;

expr : ... // <-- Original grammar of `expr`.
     | SELF '(' maybe_exprs ')'
     | ...
     ;
```

## Semantics

When entering one of the following contexts, a Rust compiler will extend
the value namespace with `Self` which maps to the tuple constructor `fn`
in the case of tuple struct, or a constant, in the case of a unit struct:

+ inherent `impl`s where the `Self` type is a tuple or unit struct
+ `trait` `impl`s where the `Self` type is a tuple or unit struct

As a result, when referring to a tuple struct, `Self` can be legally coerced
into an `fn` pointer which accepts and returns expressions of the same type as
the function pointer `Self` is referring to accepts.

Another consequence is that `Self(p_0, .., p_n)` and `Self` become
legal patterns. This works since `TupleCtor(p_0, .., p_n)` patterns are
handled by resolving them in the value namespace and checking that they
resolve to a tuple constructor. Since by definition, `Self` referring
to a tuple struct resolves to a tuple constructor, this is OK.

## Implementation notes

As an additional check on the sanity of a Rust compiler implementation,
a well formed expression `Self(v0, v1, ..)`, must be semantically equivalent to
`Self { 0: v0, 1: v1, .. }` and must also be permitted when the latter would.
Likewise the pattern `Self(p0, p1, ..)` must match exactly the same set of
values as `Self { 0: p0, 1: p1, .. }` would and must be permitted when
`Self { 0: p0, 1: p1, .. }` is well formed.

Furthermore, a well formed expression or pattern `Self` must be semantically
equivalent to `Self {}` and permitted when `Self {}` is well formed in the
same context.

For example for tuple structs, we have the typing rule:

```
Δ ⊢ τ_0  type .. Δ ⊢ τ_n  type
Δ ⊢ Self type
Γ ⊢ x_0 : τ_0 .. Γ ⊢ x_n : τ_n
Γ ⊢ Self { 0: x_0, .. n: x_n } : Self
-----------------------------------------
Γ ⊢ Self (    x_0, ..,   x_n ) : Self
```

and the operational semantics:

```
Γ ⊢ Self { 0: e_0, .., n: e_n } ⇓ v
-------------------------------------
Γ ⊢ Self {    e_0, ..,    e_n } ⇓ v
```

for unit structs, the following holds:

```
Δ ⊢ Self type
Γ ⊢ Self {} : Self
-----------------------------------------
Γ ⊢ Self    : Self
```

with the operational semantics:

```
Γ ⊢ Self {} ⇓ v
-------------------------------------
Γ ⊢ Self    ⇓ v
```

## In relation to other RFCs

This RFC expands on [RFC 593] and [RFC 1647] with
respect to where the keyword `Self` is allowed.

[RFC 593]: 0593-forbid-Self-definitions.md
[RFC 1647]: 1647-allow-self-in-where-clauses.md

# Drawbacks
[drawbacks]: #drawbacks

There are potentially some, but the author could not think of any.

# Rationale and alternatives
[alternatives]: #alternatives

This is the only design that makes sense in the sense that there really
aren't any other. Potentially, `Self(v0, ..)` should only work when the
`impl`ed type is not behind a type alias. However, since structs with named
fields supports type aliases in this respect, so should tuple structs.

Not providing this feature would preserve papercuts
and unintuitive surprises for developers.

# Unresolved questions
[unresolved]: #unresolved-questions

There are none.

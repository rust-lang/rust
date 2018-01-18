- Feature Name: tuple_struct_self_ctor
- Start Date: 2017-01-18
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Tuple `struct`s can now be constructed with `Self(v1, v2, ..)`
to match how `struct`s with named fields can be constructed
using `Self { f1: v1, f2: v2, .. }`. A simple example:

```rust
struct TheAnswer(usize);

impl Default for TheAnswer {
    fn default() -> Self { Self(42) }
}
```

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
will also extend to tuple structs. Eliminating this discrepancy between tuple
structs and those with named fields will therefore have all the benefits
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

## Teaching the contents

This RFC should not require additional effort other than spreading the
news that this now is possible as well as the reference. The changes are
seen as intuitive enough that it supports what the user already assumes
should work and will probably try at some point.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Rust (now) allows usage of `Self(v0, v1, ..)` inside inherent
and trait `impl`s of tuple structs, either when mentioning the
tuple struct directly in the `impl` header, or via a type alias.

## Desugaring

When the compiler encounters the following syntactic form specified in `EBNF`:

```ebnf
SelfTupleApply ::= "Self" "(" ExprList ")" ;
ExprList ::= Expr "," Values | Expr | "" ;
```

the compiler will desugar the application by substituting `Self(v0, v1, ..)`
for `Self { 0: v0, 1: v1, .. }` and then continue on from there. The compiler
is however free to use more direct or other approaches as long as it preserves
the semantics of desugaring to `Self { 0: v0, 1: v1, .. }`.

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

The following questions should be resolved during the RFC period:

+ Are there any syntactic ambiguities?

To the author's knowledge, there are none since following fails to compile today:

```rust
fn Self(x: u8) {} // <-- an error here since Self is a keyword.

struct F(u8);
impl F { fn x() { Self(0) } }
```
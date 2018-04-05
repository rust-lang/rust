- Feature Name: `type_alias_enum_variants`
- Start Date: 2018-02-15
- RFC PR: [rust-lang/rfcs#2338](https://github.com/rust-lang/rfcs/pull/2338)
- Rust Issue: [rust-lang/rust#49683](https://github.com/rust-lang/rust/issues/49683)

# Summary
[summary]: #summary

This RFC proposes to allow access to enum variants through type aliases. This
enables better abstraction/information hiding by encapsulating enums in aliases
without having to create another enum type and requiring the conversion from
and into the "alias" enum.

# Motivation
[motivation]: #motivation

While type aliases provide a useful means of encapsulating a type definition in
order to hide implementation details or provide a more ergonomic API, the
substitution principle currently falls down in the face of enum variants. It's
reasonable to expect that a type alias can fully replace the original type
specification, and so the lack of working support for aliased enum variants
represents an ergonomic gap in the language/type system. This can be useful in
exposing an interface from a dependency to library users while "hiding" the exact
implementation details. There's at least some evidence that people have asked
about this capability before.

Since `Self` also works as an alias, this should also enable the use of `Self`
in more places.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

In general, the simple explanation here is that type aliases can be used in
more places where you currently have to go through the original type definition,
as it relates to enum variants. As much as possible, enum variants should work
as if the original type was specified rather than the alias. This should make
type aliases easier to learn than before, because there are fewer exceptions
to their applicability.

```rust
enum Foo {
    Bar(i32),
    Baz { i: i32 },
}

type Alias = Foo;

fn main() {
    let t = Alias::Bar(0);
    let t = Alias::Baz { i: 0 };
    match t {
        Alias::Bar(_i) => {}
        Alias::Baz { i: _i } => {}
    }
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

If a path refers into an alias, the behavior for enum variants should be as
if the alias was substituted with the original type. Here are some examples of
the new behavior in edge cases:

```rust
type Alias<T> = Option<T>;

mod foo {
    pub use Alias::Some;
}

Option::<u8>::None // Not allowed
Option::None::<u8> // Ok
Alias::<u8>::None // Not allowed
Alias::None::<u8> // Ok
foo::Some::<u8> // Ok
```

This is the proposed handling for how to propagate type arguments from alias
paths:

* If the previous segment is a type (alias or enum), the variant segment
  "gifts" its arguments to that previous segment.
* If the previous segment is not a type (for example, a module), the variant
  segment treats the arguments as arguments for the variant's enum.
* In paths that specify both the alias and the variant, type arguments must
  be specified after the variant, not after the aliased type. This extends the
  current behavior to enum aliases.

# Drawbacks
[drawbacks]: #drawbacks

We should not do this if the edge cases make the implemented behavior too
complex or surprising to reason about the alias substitution.

# Rationale and alternatives
[alternatives]: #alternatives

This design seems like a straightforward extension of what type aliases are
supposed to be for. In that sense, the main alternative seems to be to do
nothing. Currently, there are two ways to work around this:

1. Require the user to implement wrapper `enum`s instead of using aliases.
   This hides more information, so it may provide more API stability. On the
   other hand, it also mandates boxing and unboxing which has a run-time
   performance cost; and API stability is already up to the user in most other
   cases.

2. Renaming of types via `use` statements. This provides a good solution in the
   case where there are no type variables that you want to fill in as part of
   the alias, but filling in variables is part of the motivating use case for
   having aliases.

As such, not implementing aliased enum variants this makes it harder to
encapsulate or hide parts of an API.

# Unresolved questions
[unresolved]: #unresolved-questions

As far as I know, there are no unresolved questions at this time.

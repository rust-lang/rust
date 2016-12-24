- Feature Name: Trait alias
- Start Date: 2016-08-31
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Traits can be aliased the same way types can be aliased with the `type` keyword.

# Motivation
[motivation]: #motivation

Sometimes, some traits are defined with parameters. For instance:

```rust
trait Foo<T> {
  // ...
}
```

It’s not uncommon to do that in *generic* crates and implement them in *backend* crates, where the
`T` template parameter gets substituted with a *backend* type.

If someone wants to write `Foo` code and be compatible with all backends, then they will use the
*generic* crate’s `Foo<_>`. However, if someone wants to freeze the backend and keep using the same
one for the whole project but want to keep the ease of backend switching, a good practice is that
the backends should exporte a specific version of the trait so that it’s possible to use `Foo`
instead of the more explicit and unwanted `Foo<BackendType>`.

# Detailed design
[design]: #detailed-design

The idea is to add a new keyword or construct for enabling trait aliasing. One shouldn’t use the
`type` keyword as a trait is not a type and that could be very confusing.

The `trait TraitAlias = Trait` was adopted as the syntax for aliasing. It creates a new trait alias
`TraitAlias` that will resolve to `Trait`.

```rust
trait TraitAlias = Debug;
```

Optionnaly, if needed, one can provide a `where` clause to express *bounds*:

```rust
trait TraitAlias = Debug where Self: Default;
```

Trait aliasing to combinations of traits is also provided with the standard `+` construct:

```rust
trait TraitAlias = Debug + Default; // same as the example above
```

Trait aliases can be used in any place arbitrary bounds would be syntactically legal. However, you
cannot use them in `impl` place but can have them as *trait objects*, in *where-clauses* and *type
parameters declarations* of course.

# Drawbacks
[drawbacks]: #drawbacks

The syntax `trait TraitAlias as Trait` makes parsers need a lookhead (`=` or `as`?).

# Alternatives
[alternatives]: #alternatives

A keyword was planned, like `alias`:

```
alias Foo = gen::Foo<Bck0>;
```

However, it’s not a good idea as it might clash with already used `alias` in codebases.

# Unresolved questions
[unresolved]: #unresolved-questions

The syntax `trait TraitAlias as Trait` is not yet stabilized and needs to be discussed.

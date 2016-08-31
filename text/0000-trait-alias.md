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

```
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

The `trait TraitAlias as Trait` is suggested as a starter construct for the discussion.

```
mod gen {
  trait Foo<T> { }
}

mod backend_0 {
  struct Bck0 {}

  trait Foo as gen::Foo<Bck0>;
}
```

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

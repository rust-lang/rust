- Start Date: 2015-01-18
- RFC PR: [rust-lang/rfcs#593](https://github.com/rust-lang/rfcs/pull/593)
- Rust Issue: [rust-lang/rust#22137](https://github.com/rust-lang/rust/issues/22137)

# Summary

Make `Self` a keyword.

# Motivation

Right now, `Self` is just a regular identifier that happens to get a special meaning
inside trait definitions and impls. Specifically, users are not forbidden from defining
a type called `Self`, which can lead to weird situations:

```rust
struct Self;

struct Foo;

impl Foo {
    fn foo(&self, _: Self) {}
}
```

This piece of code defines types called `Self` and `Foo`,
and a method `foo()` that because of the special meaning of `Self` has
the signature `fn(&Foo, Foo)`.

So in this case it is not possible to define a method on `Foo` that takes the
actual type `Self` without renaming it or creating a renamed alias.

It would also be highly unidiomatic to actually name the type `Self`
for a custom type, precisely because of this ambiguity, so preventing it outright seems like the right thing to do.

Making the identifier `Self` an keyword would prevent this situation because the user could not use it freely for custom definitions.

# Detailed design

Make the identifier `Self` a keyword that is only legal to use inside a trait definition or impl to refer to the `Self` type.

# Drawbacks

It might be unnecessary churn because people already don't run into this
in practice.

# Alternatives

Keep the status quo. It isn't a problem in practice, and just means
`Self` is the special case of a contextual type definition in the language.

# Unresolved questions

None so far

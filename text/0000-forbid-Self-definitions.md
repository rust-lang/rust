- Start Date: 2015-01-18
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Forbid the identifier `Self` for definitions.

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

Apart from that, it would also be highly unidiomatic to actually name the type `Self`
for a custom type, so preventing it outright does not seem like an issue.

Because the current situation does not cause broken code,
only unexpected type errors at most, its not really necessary to make
such a definition an hard error, so anything from making `Self` a keyword
to adding a warn-only lint would improve the situation.

# Detailed design

Implement either of:

- Make `Self` a keyword.
- Make a definition using the the identifier `Self` a hard error.
- Make a definition using the the identifier `Self` a hard warning.
- Add a error-per default lint about definitions named `Self`
- Add a warn-per default lint about definitions named `Self`

# Drawbacks

It might be unnecessary churn because people already don't run into this
in practice.

# Alternatives

Keep the status quo. It isn't a problem in practice, and just means
`Self` is the special case of a contextual type definition in the language.

# Unresolved questions

None so far

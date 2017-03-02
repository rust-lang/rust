- Start Date: 2014-03-31
- RFC PR: [rust-lang/rfcs#26](https://github.com/rust-lang/rfcs/pull/26)
- Rust Issue: [rust-lang/rust#13535](https://github.com/rust-lang/rust/issues/13535)

# Summary

This RFC is a proposal to remove the usage of the keyword `priv` from the Rust
language.

# Motivation

By removing `priv` entirely from the language, it significantly simplifies the
privacy semantics as well as the ability to explain it to newcomers. The one
remaining case, private enum variants, can be rewritten as such:

```rust
// pub enum Foo {
//     Bar,
//     priv Baz,
// }

pub enum Foo {
    Bar,
    Baz(BazInner)
}

pub struct BazInner(());

// pub enum Foo2 {
//     priv Bar2,
//     priv Baz2,
// }

pub struct Foo2 {
    variant: FooVariant
}

enum FooVariant {
    Bar2,
    Baz2,
}
```

Private enum variants are a rarely used feature of the language, and are
generally not regarded as a strong enough feature to justify the `priv` keyword
entirely.

# Detailed design

There remains only one use case of the `priv` visibility qualifier in the Rust
language, which is to make enum variants private. For example, it is possible
today to write a type such as:

```rust
pub enum Foo {
    Bar,
    priv Baz
}
```

In this example, the variant `Bar` is public, while the variant `Baz` is
private. This RFC would remove this ability to have private enum variants.

In addition to disallowing the `priv` keyword on enum variants, this RFC would
also forbid visibility qualifiers in front of enum variants entirely, as they no
longer serve any purpose.

### Status of the identifier `priv`

This RFC would demote the identifier `priv` from being a keyword to being a
reserved keyword (in case we find a use for it in the future).

# Alternatives

* Allow private enum variants, as-is today.
* Add a new keyword for `enum` which means "my variants are all private" with
  controls to make variants public.

# Unresolved questions

* Is the assertion that private enum variants are rarely used true? Are there
  legitimate use cases for keeping the `priv` keyword?

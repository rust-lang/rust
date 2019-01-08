# `type_alias_enum_variants`

The tracking issue for this feature is: [#49683]

[#49683]: https://github.com/rust-lang/rust/issues/49683

------------------------

The `type_alias_enum_variants` feature enables the use of variants on type
aliases that refer to enums, as both a constructor and a pattern. That is,
it allows for the syntax `EnumAlias::Variant`, which behaves exactly the same
as `Enum::Variant` (assuming that `EnumAlias` is an alias for some enum type
`Enum`).

Note that since `Self` exists as a type alias, this feature also enables the
use of the syntax `Self::Variant` within an impl block for an enum type.

```rust
#![feature(type_alias_enum_variants)]

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

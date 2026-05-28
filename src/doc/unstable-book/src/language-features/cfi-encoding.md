# `cfi_encoding`

The tracking issue for this feature is: [#89653]

[#89653]: https://github.com/rust-lang/rust/issues/89653

------------------------

The `cfi_encoding` feature allows the user to define a CFI encoding for a type.
It allows the user to use a different names for types that otherwise would be
required to have the same name as used in externally defined C functions.

## Examples

```rust
#![feature(cfi_encoding, extern_types)]

#[cfi_encoding = "3Foo"]
pub struct Type1(i32);

extern {
    #[cfi_encoding = "3Bar"]
    type Type2;
}
```

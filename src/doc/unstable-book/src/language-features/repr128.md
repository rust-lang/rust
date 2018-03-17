# `repr128`

The tracking issue for this feature is: [#35118]

[#35118]: https://github.com/rust-lang/rust/issues/35118

------------------------

The `repr128` feature adds support for `#[repr(u128)]` on `enum`s.

```rust
#![feature(repr128)]

#[repr(u128)]
enum Foo {
    Bar(u64),
}
```

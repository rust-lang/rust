# `concat_idents`

The tracking issue for this feature is: [#29599]

[#29599]: https://github.com/rust-lang/rust/issues/29599

------------------------

> This feature is expected to be superseded by [`macro_metavar_expr_concat`](../language-features/macro-metavar-expr-concat.md).

The `concat_idents` feature adds a macro for concatenating multiple identifiers
into one identifier.

## Examples

```rust
#![feature(concat_idents)]

fn main() {
    fn foobar() -> u32 { 23 }
    let f = concat_idents!(foo, bar);
    assert_eq!(f(), 23);
}
```

# `non_ascii_idents`

The tracking issue for this feature is: [#28979]

[#28979]: https://github.com/rust-lang/rust/issues/28979

------------------------

The `non_ascii_idents` feature adds support for non-ASCII identifiers.

## Examples

```rust
#![feature(non_ascii_idents)]

const ε: f64 = 0.00001f64;
const Π: f64 = 3.14f64;
```
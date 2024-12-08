# `cfg_boolean_literals`

The tracking issue for this feature is: [#131204]

[#131204]: https://github.com/rust-lang/rust/issues/131204

------------------------

The `cfg_boolean_literals` feature makes it possible to use the `true`/`false`
literal as cfg predicate. They always evaluate to true/false respectively.

## Examples

```rust
#![feature(cfg_boolean_literals)]

#[cfg(true)]
const A: i32 = 5;

#[cfg(all(false))]
const A: i32 = 58 * 89;
```

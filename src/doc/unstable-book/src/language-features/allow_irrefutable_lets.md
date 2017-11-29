# `irrefutable_let_pattern`

The tracking issue for this feature is: [#44495]

[#44495]: https://github.com/rust-lang/rust/issues/44495

------------------------

This feature changes the way that the irrefutable pattern is handled
in the `if let` and `while let` forms. The old way was to always error
but now with a tag the error-by-default lint can be switched off.

```rust
#![feature(irrefutable_let_pattern)]

fn main() {
    #[allow(irrefutable_let_pattern)]
    if let _ = 5 {}

    #[allow(irrefutable_let_pattern)]
    while let _ = 5 {}
}
```

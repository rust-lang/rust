# `compile_error`

The tracking issue for this feature is: [#40872]

[#40872]: https://github.com/rust-lang/rust/issues/40872

------------------------

The `compile_error` feature adds a macro which will generate a compilation
error with the specified error message.

## Examples

```rust
#![feature(compile_error)]

fn main() {
    compile_error!("The error message"); //ERROR The error message
}
```

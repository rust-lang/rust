# `abi_vectorcall`

The tracking issue for this feature is: [#124485]

[#124485]: https://github.com/rust-lang/rust/issues/124485

------------------------

Adds support for the Windows `"vectorcall"` ABI, the equivalent of `__vectorcall` in MSVC.

```rust,ignore (only-windows-or-x86-or-x86-64)
extern "vectorcall" {
    fn add_f64s(x: f64, y: f64) -> f64;
}

fn main() {
    println!("{}", add_f64s(2.0, 4.0));
}
```

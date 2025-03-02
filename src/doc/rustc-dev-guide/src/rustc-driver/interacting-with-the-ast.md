# Example: Type checking through `rustc_driver`

[`rustc_driver`] allows you to interact with Rust code at various stages of compilation.

## Getting the type of an expression

To get the type of an expression, use the [`after_analysis`] callback to get a [`TyCtxt`].

```rust
{{#include ../../examples/rustc-driver-interacting-with-the-ast.rs}}
```
[`after_analysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/trait.Callbacks.html#method.after_analysis
[`rustc_driver`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html

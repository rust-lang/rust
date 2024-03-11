# Example: Type checking through `rustc_interface`

The [`rustc_interface`] allows you to interact with Rust code at various stages of compilation.

## Getting the type of an expression

To get the type of an expression, use the [`global_ctxt`] query to [get] a [`TyCtxt`].
The following was tested with <!-- date-check: jan 2024 --> `nightly-2024-01-19`:

```rust
{{#include ../examples/rustc-driver-interacting-with-the-ast.rs}}
```
[get]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.GlobalCtxt.html#method.enter
[`global_ctxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/queries/struct.Queries.html#method.global_ctxt
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html

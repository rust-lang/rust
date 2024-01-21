# Example: Type checking through `rustc_interface`

`rustc_interface` allows you to interact with Rust code at various stages of compilation.

## Getting the type of an expression

To get the type of an expression, use the `global_ctxt` to get a `TyCtxt`.
The following was tested with <!-- date-check: jan 2024 --> `nightly-2024-01-19`:

```rust
{{#include ../examples/rustc-driver-interacting-with-the-ast.rs}}
```

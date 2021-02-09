# `codegen-backend`

The tracking issue for this feature is: [#77933](https://github.com/rust-lang/rust/issues/77933).

------------------------

This feature allows you to specify a path to a dynamic library to use as rustc's
code generation backend at runtime.

Set the `-Zcodegen-backend=<path>` compiler flag to specify the location of the
backend. The library must be of crate type `dylib` and must contain a function
named `__rustc_codegen_backend` with a signature of `fn() -> Box<dyn rustc_codegen_ssa::traits::CodegenBackend>`.

## Example
See also the [`hotplug_codegen_backend`](https://github.com/rust-lang/rust/tree/master/src/test/run-make-fulldeps/hotplug_codegen_backend) test
for a full example.

```rust,ignore
use rustc_codegen_ssa::traits::CodegenBackend;

struct MyBackend;

impl CodegenBackend for MyBackend {
   // Implement codegen methods
}

#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(MyBackend)
}
```

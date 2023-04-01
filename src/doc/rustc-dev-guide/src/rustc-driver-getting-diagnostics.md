# Example: Getting diagnostic through `rustc_interface`

`rustc_interface` allows you to intercept diagnostics that would otherwise be printed to stderr.

## Getting diagnostics

To get diagnostics from the compiler,
configure `rustc_interface::Config` to output diagnostic to a buffer,
and run `TyCtxt.analysis`. The following was tested
with <!-- date-check: mar 2023 --> `nightly-2023-03-27`:

```rust
{{#include ../examples/rustc-driver-getting-diagnostics.rs}}
```

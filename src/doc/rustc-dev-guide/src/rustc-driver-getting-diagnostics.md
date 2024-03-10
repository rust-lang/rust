# Example: Getting diagnostic through `rustc_interface`

The [`rustc_interface`] allows you to intercept diagnostics that would
otherwise be printed to stderr.

## Getting diagnostics

To get diagnostics from the compiler,
configure [`rustc_interface::Config`] to output diagnostic to a buffer,
and run [`TyCtxt.analysis`]. The following was tested
with <!-- date-check: jan 2024 --> `nightly-2024-01-19`:

```rust
{{#include ../examples/rustc-driver-getting-diagnostics.rs}}
```

[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`rustc_interface::Config`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Config.html
[`TyCtxt.analysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/passes/fn.analysis.html
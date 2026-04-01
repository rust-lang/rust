# Example: Getting diagnostic through `rustc_interface`

The [`rustc_interface`] allows you to intercept diagnostics that would
otherwise be printed to stderr.

## Getting diagnostics

To get diagnostics from the compiler,
configure [`rustc_interface::Config`] to output diagnostic to a buffer,
and run [`rustc_hir_typeck::typeck`] for each item.

```rust
{{#include ../../examples/rustc-interface-getting-diagnostics.rs}}
```

[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[`rustc_interface::Config`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Config.html
[`TyCtxt.analysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/passes/fn.analysis.html
[`rustc_hir_typeck::typeck`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn.typeck.html

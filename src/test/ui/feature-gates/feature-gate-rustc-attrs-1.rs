// ignore-tidy-linelength

// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#[rustc_variance] //~ ERROR the `#[rustc_variance]` attribute is just used for rustc unit tests and will never be stable
#[rustc_error] //~ ERROR the `#[rustc_error]` attribute is just used for rustc unit tests and will never be stable
#[rustc_nonnull_optimization_guaranteed] //~ ERROR the `#[rustc_nonnull_optimization_guaranteed]` attribute is just used to enable niche optimizations in libcore and will never be stable

fn main() {}

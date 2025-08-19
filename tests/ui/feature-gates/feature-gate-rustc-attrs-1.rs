// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#[rustc_variance]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `#[rustc_variance]` attribute is an internal implementation detail that will never be stable
//~| NOTE the `#[rustc_variance]` attribute is used for rustc unit tests
#[rustc_nonnull_optimization_guaranteed]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `#[rustc_nonnull_optimization_guaranteed]` attribute is an internal implementation detail that will never be stable
//~| NOTE  the `#[rustc_nonnull_optimization_guaranteed]` attribute is just used to document guaranteed niche optimizations in the standard library
//~| NOTE the compiler does not even check whether the type indeed is being non-null-optimized; it is your responsibility to ensure that the attribute is only used on types that are optimized
fn main() {}

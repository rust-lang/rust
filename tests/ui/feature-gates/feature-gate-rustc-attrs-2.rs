// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#[rustc_dump_variances]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `rustc_dump_variances` attribute is an internal implementation detail that will never be stable
//~| NOTE the `rustc_dump_variances` attribute is used for rustc unit tests
enum E {}

fn main() {}

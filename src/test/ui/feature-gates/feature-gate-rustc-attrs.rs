// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#[rustc_dummy]
//~^ ERROR used by the test suite

fn main() {}

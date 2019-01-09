// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#[rustc_foo]
//~^ ERROR unless otherwise specified, attributes with the prefix `rustc_` are reserved

fn main() {}

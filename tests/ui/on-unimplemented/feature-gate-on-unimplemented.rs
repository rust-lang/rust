// Test that `#[rustc_on_unimplemented]` is gated by `rustc_attrs` feature gate.

#![allow(malformed_diagnostic_filters)]
//~^ WARN unknown lint: `malformed_diagnostic_filters`
//~| NOTE the `malformed_diagnostic_filters` lint is unstable
//~| HELP add `#![feature(rustc_attrs)]` to the crate attributes to enable
//~| NOTE `#[warn(unknown_lints)]` on by default

#[rustc_on_unimplemented(label = "test error `{Self}` with `{Bar}`")]
//~^ ERROR use of an internal attribute [E0658]
//~| HELP add `#![feature(rustc_attrs)]` to the crate attributes to enable
//~| NOTE the `rustc_on_unimplemented` attribute is an internal implementation detail that will never be stable
//~| NOTE see the `diagnostic::on_unimplemented` attribute for the stable equivalent of this attribute
trait Foo<Bar> {}

fn main() {}

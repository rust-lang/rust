#[rustc_diagnostic_item = "foomp"]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `rustc_diagnostic_item` attribute is an internal implementation detail that will never be stable
//~| NOTE the `rustc_diagnostic_item` attribute allows the compiler to reference types from the standard library for diagnostic purposes
struct Foomp;
fn main() {}

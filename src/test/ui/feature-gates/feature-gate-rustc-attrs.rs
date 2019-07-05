// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#![feature(decl_macro)]

mod rustc { pub macro unknown() {} }
mod unknown { pub macro rustc() {} }

#[rustc::unknown]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR macro `rustc::unknown` may not be used in attributes
fn f() {}

#[unknown::rustc]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR macro `unknown::rustc` may not be used in attributes
fn g() {}

#[rustc_dummy]
//~^ ERROR used by the test suite
#[rustc_unknown]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR attribute `rustc_unknown` is currently unknown
fn main() {}

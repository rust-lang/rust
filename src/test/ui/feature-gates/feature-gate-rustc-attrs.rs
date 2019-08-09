// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#![feature(decl_macro)]

mod rustc { pub macro unknown() {} }
mod unknown { pub macro rustc() {} }

#[rustc::unknown]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR expected attribute, found macro `rustc::unknown`
fn f() {}

#[unknown::rustc]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR expected attribute, found macro `unknown::rustc`
fn g() {}

#[rustc_dummy]
//~^ ERROR used by the test suite
#[rustc_unknown]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR cannot find attribute macro `rustc_unknown` in this scope
fn main() {}

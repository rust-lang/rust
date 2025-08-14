// Test that `#[rustc_*]` attributes are gated by `rustc_attrs` feature gate.

#![feature(decl_macro)]

mod rustc { pub macro unknown() {} }
mod unknown { pub macro rustc() {} }

#[rustc::unknown]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR expected attribute, found macro `rustc::unknown`
//~| NOTE not an attribute
fn f() {}

#[unknown::rustc]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR expected attribute, found macro `unknown::rustc`
//~| NOTE not an attribute
fn g() {}

#[rustc_dummy]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `#[rustc_dummy]` attribute is an internal implementation detail that will never be stable
//~| NOTE the `#[rustc_dummy]` attribute is used for rustc unit tests
#[rustc_unknown]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR cannot find attribute `rustc_unknown` in this scope
fn main() {}

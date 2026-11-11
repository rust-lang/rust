//@ aux-build:cross-crate-static-reference-and-field-access.rs
//@ check-pass

//! Regression test for https://github.com/rust-lang/rust/issues/29265
//! Exposed an bug where referencing a static variable from another crate
//! was causing an error.

extern crate cross_crate_static_reference_and_field_access as lib;

static _UNUSED: &'static lib::SomeType = &lib::SOME_VALUE;

fn main() {
    vec![0u8; lib::SOME_VALUE.some_member];
}

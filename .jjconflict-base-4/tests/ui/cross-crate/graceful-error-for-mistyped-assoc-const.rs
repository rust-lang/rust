//@ aux-build:graceful-error-for-mistyped-assoc-const.rs
//! Regression test for https://github.com/rust-lang/rust/issues/41549
//! Mismatched types on exernal associated constants used to ice,
//! this test confirms that it errors correctly.
extern crate graceful_error_for_mistyped_assoc_const as issue_41549;

struct S;

impl issue_41549::Trait for S {
    const CONST: () = (); //~ ERROR incompatible type for trait
}

fn main() {}

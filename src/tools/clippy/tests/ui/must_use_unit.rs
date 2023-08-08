//@run-rustfix
//@aux-build:proc_macros.rs:proc-macro

#![warn(clippy::must_use_unit)]
#![allow(clippy::unused_unit)]

extern crate proc_macros;
use proc_macros::external;

#[must_use]
pub fn must_use_default() {}

#[must_use]
pub fn must_use_unit() -> () {}

#[must_use = "With note"]
pub fn must_use_with_note() {}

fn main() {
    must_use_default();
    must_use_unit();
    must_use_with_note();

    // We should not lint in external macros
    external!(
        #[must_use]
        fn foo() {}
    );
}

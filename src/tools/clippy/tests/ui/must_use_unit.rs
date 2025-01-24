//@aux-build:proc_macros.rs

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

#[cfg_attr(all(), must_use, deprecated)]
fn issue_12320() {}

#[cfg_attr(all(), deprecated, doc = "foo", must_use)]
fn issue_12320_2() {}

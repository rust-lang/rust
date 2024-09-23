//@ build-fail
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]

// Test that forced inlining into closures w/ errors works as expected.

#[rustc_no_mir_inline]
#[rustc_force_inline]
pub fn callee() {
}

#[rustc_no_mir_inline]
#[rustc_force_inline = "the test requires it"]
pub fn callee_justified() {
}

pub fn caller() {
    (|| {
        callee();
//~^ ERROR `callee` could not be inlined into `caller::{closure#0}` but is required to be inlined

        callee_justified();
//~^ ERROR `callee_justified` could not be inlined into `caller::{closure#0}` but is required to be inlined
    })();
}

//@ check-fail
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]

// Test that forced inlining w/ errors works as expected.

#[rustc_no_mir_inline]
#[rustc_force_inline]
//~^ ERROR `callee` is incompatible with `#[rustc_force_inline]`
pub fn callee() {
}

#[rustc_no_mir_inline]
#[rustc_force_inline = "the test requires it"]
//~^ ERROR `callee_justified` is incompatible with `#[rustc_force_inline]`
pub fn callee_justified() {
}

pub fn caller() {
    callee();
    callee_justified();
}

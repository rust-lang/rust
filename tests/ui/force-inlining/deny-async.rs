//@ compile-flags: --crate-type=lib
//@ edition: 2021

#![allow(internal_features)]
#![feature(rustc_attrs)]

// Test that forced inlining into async functions w/ errors works as expected.

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

async fn async_caller() {
    callee();
    //~^ ERROR `callee` could not be inlined
    callee_justified();
    //~^ ERROR `callee_justified` could not be inlined
}

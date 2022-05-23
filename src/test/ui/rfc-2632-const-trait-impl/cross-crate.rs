// revisions: stock gated
#![cfg_attr(gated, feature(const_trait_impl))]

// aux-build: cross-crate.rs
extern crate cross_crate;

use cross_crate::*;

fn non_const_context() {
    NonConst.func();
    Const.func();
}

const fn const_context() {
    NonConst.func(); //~ ERROR: cannot call non-const fn
    //[gated]~^ ERROR: the trait bound
    Const.func();
    //[stock]~^ ERROR: cannot call non-const fn
}

fn main() {}

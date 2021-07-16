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
    NonConst.func();
    //~^ ERROR: calls in constant functions are limited to constant functions, tuple structs and tuple variants
    Const.func();
    //[stock]~^ ERROR: calls in constant functions are limited to constant functions, tuple structs and tuple variants
}

fn main() {}

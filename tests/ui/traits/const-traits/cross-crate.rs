//@ revisions: stock gated stocknc gatednc
//@ [gated] check-pass
//@ compile-flags: -Znext-solver
#![cfg_attr(any(gated, gatednc), feature(const_trait_impl))]
#![allow(incomplete_features)]

//@ aux-build: cross-crate.rs
extern crate cross_crate;

use cross_crate::*;

fn non_const_context() {
    NonConst.func();
    Const.func();
}

const fn const_context() {
    #[cfg(any(stocknc, gatednc))]
    NonConst.func();
    //[stocknc]~^ ERROR: cannot call
    //[gatednc]~^^ ERROR: the trait bound
    Const.func();
    //[stock,stocknc]~^ ERROR: cannot call
}

fn main() {}

//@ compile-flags: -Znext-solver
//@ check-fail

// A wrap produces a new value, not a place, so assigning to it is error.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn f(x: u8, y: unsafe<> u8) {
    unsafe { wrap_binder!(x) = y };
    //~^ ERROR type annotations needed
    //~| ERROR invalid left-hand side of assignment
}

fn g(x: u8, y: unsafe<> u8) {
    unsafe { wrap_binder!(x; unsafe<> u8) = y };
    //~^ ERROR type ascription
    //~| ERROR invalid left-hand side of assignment
}

fn h(mut b: unsafe<> u8) -> unsafe<> u8 {
    unsafe { unwrap_binder!(b) = 1 };
    b
}

fn main() {}

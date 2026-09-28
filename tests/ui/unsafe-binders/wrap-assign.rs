//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

// A wrap produces a new value, not a place, so assigning to it is error.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::wrap_binder;

fn f(x: u8, y: unsafe<> u8) {
    unsafe { wrap_binder!(x; unsafe<> u8) = y };
}

fn main() {}

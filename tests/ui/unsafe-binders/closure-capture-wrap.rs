//@ compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2021

// Wrapping a captured value in a close is treated as a normal use.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;
use std::unsafe_binder::wrap_binder;

fn capture_copy(x: u8) -> (unsafe<> u8, u8) {
    let f = || -> unsafe<> u8 { unsafe { wrap_binder!(x) } };
    (f(), x)
}

fn capture_move(x: MD<String>) -> unsafe<> MD<String> {
    let f = || -> unsafe<> MD<String> { unsafe { wrap_binder!(x) } };
    f()
}

fn main() {}

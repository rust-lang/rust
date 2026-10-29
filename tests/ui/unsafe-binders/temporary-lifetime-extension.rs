//@ compile-flags: -Znext-solver

// Temporary lifetime extension doesn't go through `unwrap_binder!`.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn mk() -> unsafe<> (u8, u8) {
    unsafe { wrap_binder!((1, 2)) }
}

fn through_unwrap() {
    let r = unsafe { &unwrap_binder!(mk()).1 };
    //~^ ERROR temporary value dropped while borrowed
    let _ = *r;
}

fn through_field() {
    let r = &(1u8, 2u8).1;
    let _ = *r;
}

fn main() {}

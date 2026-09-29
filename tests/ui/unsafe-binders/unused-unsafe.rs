//@ compile-flags: -Znext-solver

// The unsafe block around a wrap or unwrap counts as used.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]
#![deny(unused_unsafe)]

use std::unsafe_binder::unwrap_binder;

fn used(b: unsafe<> u8) -> u8 {
    unsafe { unwrap_binder!(b) }
}

fn nested(b: unsafe<> u8) -> u8 {
    unsafe { unsafe { unwrap_binder!(b) } }
    //~^ ERROR unnecessary `unsafe` block
}

fn main() {}

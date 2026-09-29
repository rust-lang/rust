//@ compile-flags: -Znext-solver

// Free lifetimes in an unsafe binder keep the variance they have in the inner type.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::cell::Cell;

fn covariant<'x>(b: unsafe<'a> (&'a u8, &'static u8)) -> unsafe<'a> (&'a u8, &'x u8) {
    b
}

fn covariant_wrong_way<'x>(b: unsafe<'a> (&'a u8, &'x u8)) -> unsafe<'a> (&'a u8, &'static u8) {
    b
    //~^ ERROR lifetime may not live long enough
}

fn invariant<'x>(
    b: unsafe<'a> (&'a u8, &'x Cell<&'static u8>),
) -> unsafe<'a> (&'a u8, &'x Cell<&'x u8>) {
    b
    //~^ ERROR lifetime may not live long enough
}

fn main() {}

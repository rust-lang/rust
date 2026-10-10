//@ compile-flags: -Znext-solver
//@ check-fail

// Unsafe binders are impl neither `Copy` nor `Clone`.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;

fn twice(b: unsafe<'a> &'a u8) {
    let _c = b;
    let _d = b;
    //~^ ERROR use of moved value
}

fn clone(b: unsafe<'a> &'a u8) {
    let _ = Clone::clone(&b);
    //~^ ERROR the trait bound
}

union U {
    a: unsafe<'a> &'a u8,
    //~^ ERROR field must implement `Copy`
}

fn nested(_: unsafe<'a> (&'a u8, unsafe<'b> &'b u8)) {}
//~^ ERROR the trait bound

fn nested_md(_: unsafe<'a> (&'a u8, MD<unsafe<'b> &'b u8>)) {}

fn nested_outer_lifetime(_: unsafe<'a> MD<unsafe<> Vec<&'a u8>>) {}
//~^ ERROR the trait bound

#[derive(Clone, Copy)]
struct S {
    //~^ ERROR the trait `Copy` cannot be implemented for this type
    b: unsafe<'a> &'a u8,
    //~^ ERROR the trait bound
}

fn main() {}

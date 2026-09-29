//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

// Unsafe binders are impl neither `Copy` nor `Clone`.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;

fn twice(b: unsafe<'a> &'a u8) {
    let _c = b;
    let _d = b;
}

fn clone(b: unsafe<'a> &'a u8) {
    let _ = Clone::clone(&b);
}

union U {
    a: unsafe<'a> &'a u8,
}

fn nested(_: unsafe<'a> (&'a u8, unsafe<'b> &'b u8)) {}

fn nested_md(_: unsafe<'a> (&'a u8, MD<unsafe<'b> &'b u8>)) {}

fn nested_outer_lifetime(_: unsafe<'a> MD<unsafe<> Vec<&'a u8>>) {}

#[derive(Clone, Copy)]
struct S {
    b: unsafe<'a> &'a u8,
}

fn main() {}

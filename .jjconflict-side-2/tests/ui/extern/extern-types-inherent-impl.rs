// Test that inherent impls can be defined for extern types.

//@ check-pass
//@ aux-build:extern-types-inherent-impl.rs

#![feature(extern_types)]

extern crate extern_types_inherent_impl;
use extern_types_inherent_impl::CrossCrate;

extern "C" {
    type Local;
}

impl Local {
    fn foo(&self) {}
}

fn use_foo(x: &Local, y: &CrossCrate) {
    Local::foo(x);
    x.foo();
    CrossCrate::foo(y);
    y.foo();
}

fn main() {}

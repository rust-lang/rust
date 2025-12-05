//@ check-pass
//@ aux-build:xcrate-const-ctor-a.rs

#![feature(adt_const_params)]

extern crate xcrate_const_ctor_a;
use xcrate_const_ctor_a::Foo;

fn bar<const N: Foo>() {}

fn baz() {
    bar::<{ Foo }>();
}

fn main() {}

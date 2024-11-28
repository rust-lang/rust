//@ known-bug: #132985
//@ aux-build:aux132985.rs

#![allow(incomplete_features)]
#![feature(min_generic_const_args)]
#![feature(adt_const_params)]

extern crate aux132985;
use aux132985::Foo;

fn bar<const N: Foo>() {}

fn baz() {
    bar::<{ Foo }>();
}

fn main() {}

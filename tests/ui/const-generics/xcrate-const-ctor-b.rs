//@ check-pass
//@ revisions: normal mgca
//@ aux-build:xcrate-const-ctor-a.rs

#![feature(adt_const_params)]
#![cfg_attr(mgca, feature(min_generic_const_args), allow(incomplete_features))]

extern crate xcrate_const_ctor_a;
use xcrate_const_ctor_a::Foo;

fn bar<const N: Foo>() {}

fn baz() {
    bar::<{ Foo }>();
}

fn main() {}

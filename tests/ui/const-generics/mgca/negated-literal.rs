//@ check-pass

#![feature(adt_const_params, min_generic_const_args)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo {
    field: isize
}

fn foo<const F: Foo>() {}

fn main() {
    foo::<{ Foo { field: -1 } }>();
}

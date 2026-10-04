//@run-pass
#![feature(min_adt_const_params)]

use std::marker::ConstParamTy;

#[derive(ConstParamTy, Eq, PartialEq)]
#[allow(dead_code)]
struct Foo {
    pub field: u32,
}

fn main() {}

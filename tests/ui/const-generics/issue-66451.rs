#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::UnsizedConstParamTy;

#[derive(Debug, PartialEq, Eq, UnsizedConstParamTy)]
struct Foo {
    value: i32,
    nested: &'static Bar<i32>,
}

#[derive(Debug, PartialEq, Eq, UnsizedConstParamTy)]
struct Bar<T>(T);

struct Test<const F: Foo>;

fn main() {
    let x: Test<{ Foo { value: 3, nested: &Bar(4) } }> = Test;
    let y: Test<{ Foo { value: 3, nested: &Bar(5) } }> = x; //~ ERROR mismatched types
}

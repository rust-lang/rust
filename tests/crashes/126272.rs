//@ known-bug: rust-lang/rust#126272

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Debug, PartialEq, Eq, ConstParamTy)]
struct Foo {
    value: i32,
    nested: &'static Bar<std::fmt::Debug>,
}

#[derive(Debug, PartialEq, Eq, ConstParamTy)]
struct Bar<T>(T);

struct Test<const F: Foo>;

fn main() {
    let x: Test<
        {
            Foo {
                value: 3,
                nested: &Bar(4),
            }
        },
    > = Test;
}

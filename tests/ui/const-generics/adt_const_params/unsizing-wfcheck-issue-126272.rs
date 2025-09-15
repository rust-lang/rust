// This ensures we don't ICE in situations like rust-lang/rust#126272.

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Debug, PartialEq, Eq, ConstParamTy)]
//~^ ERROR the trait `ConstParamTy_`
struct Foo {
    nested: &'static Bar<dyn std::fmt::Debug>,
    //~^ ERROR the size for values
    //~| ERROR the size for values
    //~| ERROR binary operation `==` cannot
    //~| ERROR the trait bound `dyn Debug: Eq`
    //~| ERROR the size for values
}

#[derive(Debug, PartialEq, Eq, ConstParamTy)]
struct Bar<T>(T);

struct Test<const F: Foo>;

fn main() {
    let x: Test<{ Foo { nested: &Bar(4) } }> = Test;
    //~^ ERROR the size for values
}

#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo;
struct Bar;

fn takes_array<const A: [Foo; 1]>() {}

fn main() {
    takes_array::<{ [Bar] }>();
    //~^ ERROR: the constant `Bar` is not of type `Foo`
}

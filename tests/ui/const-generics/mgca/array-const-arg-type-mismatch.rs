#![expect(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args)]
use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo;

struct Bar;

fn test<const N: [Foo; 1]>() {}

fn main() {
    test::<{ [Bar] }>();
    //~^ ERROR constant `Bar` is not of type `Foo`
}

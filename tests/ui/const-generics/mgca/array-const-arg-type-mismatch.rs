#![expect(incomplete_features)]
#![feature(adt_const_params, gca_min_const_items, gca_macroless_args)]
use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo;

struct Bar;

fn test<const N: [Foo; 1]>() {}

fn main() {
    test::<{ [Bar] }>();
    //~^ ERROR constant `Bar` is not of type `Foo`
}

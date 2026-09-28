#![feature(
    adt_const_params,
    gca_adts,
    gca_macroless_args,
    gca_min_const_items,
    unsized_const_params
)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo;

struct Bar;

fn takes_tuple_with_array<const A: ([Foo; 1], u32)>() {}

fn main() {
    takes_tuple_with_array::<{ ([Bar], 1) }>();
    //~^ ERROR the constant `Bar` is not of type `Foo`
}

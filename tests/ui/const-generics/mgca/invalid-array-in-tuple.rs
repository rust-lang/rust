#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo;

struct Bar;

fn takes_tuple_with_array<const A: ([Foo; 1], u32)>() {}

fn main() {
    takes_tuple_with_array::<{ ([Bar], 1) }>();
    //~^ ERROR the constant `Bar` is not of type `Foo`
}

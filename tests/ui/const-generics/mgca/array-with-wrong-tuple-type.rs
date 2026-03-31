#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct A;

fn takes_tuple<const N: [(u32, u32); 1]>() {}
fn takes_nested_tuple<const N: [(u32, (u32, u32)); 1]>() {}


fn main() {
    takes_tuple::<{ [A] }>();
    //~^ ERROR the constant `A` is not of type `(u32, u32)`
    takes_nested_tuple::<{ [A] }>();
    //~^ ERROR the constant `A` is not of type `(u32, (u32, u32))`
}

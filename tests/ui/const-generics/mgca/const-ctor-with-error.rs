// to ensure it does not ices like before

#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]
use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq)]
enum Option<T> {
    #[allow(dead_code)]
    Some(T),
    None,
}

fn pass_enum<const P: Option<u32>>() {}

fn main() {
    pass_enum::<{ None }>();
    //~^ ERROR  missing generics for enum `std::option::Option`
}

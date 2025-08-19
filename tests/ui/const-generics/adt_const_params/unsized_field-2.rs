//@ aux-build:unsized_const_param.rs
#![feature(adt_const_params, unsized_const_params)]
//~^ WARN: the feature `unsized_const_params` is incomplete


// TODO: remove test

extern crate unsized_const_param;


#[derive(std::marker::ConstParamTy, Eq, PartialEq)]
struct B(unsized_const_param::GenericNotUnsizedParam<&'static [u8]>);

fn main() {}

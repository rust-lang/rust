//@ aux-build:unsized_const_param.rs
#![feature(adt_const_params)]

extern crate unsized_const_param;

use std::marker::ConstParamTy;

#[derive(ConstParamTy, Eq, PartialEq)]
//~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type
struct A([u8]);

#[derive(ConstParamTy, Eq, PartialEq)]
//~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type
struct B(&'static [u8]);

#[derive(ConstParamTy, Eq, PartialEq)]
struct C(unsized_const_param::Foo);

#[derive(std::marker::ConstParamTy, Eq, PartialEq)]
//~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type
struct D(unsized_const_param::GenericNotUnsizedParam<&'static [u8]>);

fn main() {}

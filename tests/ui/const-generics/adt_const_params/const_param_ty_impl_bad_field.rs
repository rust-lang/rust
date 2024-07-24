#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

#[derive(PartialEq, Eq)]
struct NotParam;

#[derive(PartialEq, Eq)]
struct CantParam(NotParam);

impl std::marker::UnsizedConstParamTy for CantParam {}
//~^ error: the trait `ConstParamTy_` cannot be implemented for this type

#[derive(std::marker::UnsizedConstParamTy, Eq, PartialEq)]
//~^ error: the trait `ConstParamTy_` cannot be implemented for this type
struct CantParamDerive(NotParam);

fn main() {}

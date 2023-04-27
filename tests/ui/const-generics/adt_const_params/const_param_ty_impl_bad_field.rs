#![allow(incomplete_features)]
#![feature(adt_const_params)]

#[derive(PartialEq, Eq)]
struct NotParam;

#[derive(PartialEq, Eq)]
struct CantParam(NotParam);

impl std::marker::ConstParamTy for CantParam {}
//~^ error: the trait `ConstParamTy` cannot be implemented for this type

fn main() {}

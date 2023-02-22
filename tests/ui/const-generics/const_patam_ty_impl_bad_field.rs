#![feature(const_param_ty_trait)]

#[derive(PartialEq, Eq)]
struct NotParam;

#[derive(PartialEq, Eq)]
struct CantParam(NotParam);

impl std::marker::ConstParamTy for CantParam {}
//~^ error: the trait `ConstParamTy` may not be implemented for this type

fn main() {}

#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

#[derive(PartialEq, Eq)]
struct ImplementsConstParamTy;
impl std::marker::UnsizedConstParamTy for ImplementsConstParamTy {}

struct CantParam(ImplementsConstParamTy);

impl std::marker::UnsizedConstParamTy for CantParam {}
//~^ error: the type `CantParam` does not `#[derive(PartialEq)]`
//~| ERROR the trait bound `CantParam: Eq` is not satisfied

#[derive(std::marker::UnsizedConstParamTy)]
//~^ error: the type `CantParamDerive` does not `#[derive(PartialEq)]`
//~| ERROR the trait bound `CantParamDerive: Eq` is not satisfied
struct CantParamDerive(ImplementsConstParamTy);

fn check<T: std::marker::UnsizedConstParamTy>() {}

fn main() {
    check::<ImplementsConstParamTy>();
}

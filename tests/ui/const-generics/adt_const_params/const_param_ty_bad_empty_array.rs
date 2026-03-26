#![allow(incomplete_features)]
#![feature(adt_const_params, const_param_ty_trait)]

#[derive(PartialEq, Eq)]
struct NotParam;

fn check<T: std::marker::ConstParamTy_>() {}

fn main() {
    check::<[NotParam; 0]>();
    //~^ error: `NotParam` can't be used as a const parameter type
}

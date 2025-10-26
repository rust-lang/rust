#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

#[derive(PartialEq, Eq)]
struct NotParam;

fn check<T: std::marker::ConstParamTy_ + ?Sized>() {}

fn main() {
    check::<&NotParam>(); //~ error: `NotParam` can't be used as a const parameter type
    check::<[NotParam]>(); //~ error: `NotParam` can't be used as a const parameter type
    check::<[NotParam; 17]>(); //~ error: `NotParam` can't be used as a const parameter type
}

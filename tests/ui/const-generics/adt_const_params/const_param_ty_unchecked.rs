//@run-pass
//! Ensure that const_param_ty_unchecked gate allow
//! bypassing `ConstParamTy_` implementation check

#![allow(dead_code, incomplete_features)]
#![feature(const_param_ty_unchecked, const_param_ty_trait)]

use std::marker::ConstParamTy_;

struct Miow;

struct Meoww(Miow);

struct Float {
    float: f32,
}

impl ConstParamTy_ for Meoww {}
impl ConstParamTy_ for Float {}

fn something2<const N: *mut u8>() {}
fn something<const N: f64>(a: f64) -> f64 {
    N + a
}

fn main() {
    assert_eq!(2.0, something::<{1.0}>(1.0));
}

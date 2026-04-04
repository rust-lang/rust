// gate-test-min_adt_const_params
//@run-pass
#![feature(min_adt_const_params, const_param_ty_trait)]
#![allow(incomplete_features, dead_code)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
pub struct Meowl {
    pub public: i32,
    pub also_public: i32
}

impl ConstParamTy_ for Meowl {}

fn meoow<const N: Meowl>() {}

fn main() {}

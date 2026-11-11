//! Ensure we enforce `min_adt_const_params` rules on any adt `ConstParamTy_`
//! implementation unless `adt_const_params` feature is used.
#![allow(incomplete_features)]
#![feature(const_param_ty_trait)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
pub struct Fumo {
    cirno: i32,
    pub(crate) reimu: i32
}

impl ConstParamTy_ for Fumo {}
                    //~^ ERROR: the trait `ConstParamTy` may not be implemented for this struct

fn main() {}

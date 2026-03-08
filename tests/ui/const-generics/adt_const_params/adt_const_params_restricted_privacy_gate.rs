// gate-test-adt_const_params_restricted_privacy
//@run-pass
//! Without the #![feature(adt_const_params_restricted_privacy)]
//! this shouldn't fail.
#![feature(unsized_const_params, adt_const_params)]
#![allow(incomplete_features)]
use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
pub struct Meowl {
    pub public: i32,
    private: i32
}

impl ConstParamTy_ for Meowl {}

fn main() {}

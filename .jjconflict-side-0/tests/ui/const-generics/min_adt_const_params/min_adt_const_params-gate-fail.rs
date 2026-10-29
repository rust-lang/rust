//! Ensure min_adt_const_params enforce
//! struct's visibility on its fields
#![allow(incomplete_features)]
#![feature(min_adt_const_params)]
#![feature(const_param_ty_trait)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
pub struct Meowl {
    pub public: i32,
    private: i32
}

#[derive(PartialEq, Eq)]
pub struct Meowl2 {
    pub a: i32,
    pub b: i32
}

#[derive(PartialEq, Eq)]
pub(crate) struct Meowl3 {
    pub(crate) a: i32,
    pub b: i32
}

impl ConstParamTy_ for Meowl {}
                    //~^ ERROR the trait `ConstParamTy` may not be implemented for this struct
impl ConstParamTy_ for Meowl2 {}
impl ConstParamTy_ for Meowl3 {}

fn something<const N: Meowl2>() {}
fn something2<const N: Meowl3>() {}

fn main() {}

//! Ensure adt_const_params_restricted_privacy enforce
//! struct's visibility on its fields
#![allow(incomplete_features)]
#![feature(adt_const_params_restricted_privacy)]
#![feature(adt_const_params)]
#![feature(unsized_const_params)]

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
                    //~^ ERROR the trait `ConstParamTy` may not be implemented for this struct

fn main() {}

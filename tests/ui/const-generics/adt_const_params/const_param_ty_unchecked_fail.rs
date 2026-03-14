// gate-test-const_param_ty_unchecked
//! Ensure this fails when const_param_ty_unchecked isn't used

#![allow(incomplete_features)]
//FIXME: change to const_param_ty_trait gate when it will be merged
#![feature(unsized_const_params, adt_const_params)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
struct Miow; // Miow does not implement ConstParamTy_.

#[derive(PartialEq, Eq)]
struct Meoww(Miow);

impl ConstParamTy_ for Meoww {}
                    //~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type [E0204]

fn main() {}

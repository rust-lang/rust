//@check-pass
//! Ensure that const_param_ty_unchecked gate allow
//! bypassing `ConstParamTy_` implementation check

#![allow(incomplete_features)]
//FIXME: change to const_param_ty_trait gate when it will be merged
#![feature(unsized_const_params, adt_const_params)]
#![feature(const_param_ty_unchecked)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
struct Miow; // Miow does not implement ConstParamTy_.

#[derive(PartialEq, Eq)]
struct Meoww(Miow);

impl ConstParamTy_ for Meoww {}

fn main() {}

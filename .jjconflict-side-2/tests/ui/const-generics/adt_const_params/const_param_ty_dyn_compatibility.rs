#![feature(adt_const_params, const_param_ty_trait)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn foo(a: &dyn ConstParamTy_) {}
//~^ ERROR: the trait `ConstParamTy_`

fn main() {}

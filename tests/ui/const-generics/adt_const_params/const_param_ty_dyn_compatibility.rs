#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn foo(a: &dyn ConstParamTy_) {}
//~^ ERROR: the trait `ConstParamTy_`

fn main() {}

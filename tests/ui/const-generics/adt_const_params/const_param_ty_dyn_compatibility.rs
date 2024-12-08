#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::{ConstParamTy_, UnsizedConstParamTy};

fn foo(a: &dyn ConstParamTy_) {}
//~^ ERROR: the trait `ConstParamTy_`

fn bar(a: &dyn UnsizedConstParamTy) {}
//~^ ERROR: the trait `UnsizedConstParamTy`

fn main() {}

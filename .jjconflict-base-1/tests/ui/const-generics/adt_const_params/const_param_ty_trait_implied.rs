//@check-pass
#![feature(unsized_const_params, adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn miow<T: ConstParamTy_>() {}

fn main() {}

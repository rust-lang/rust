//@check-pass
#![feature(const_param_ty_trait)]

use std::marker::ConstParamTy_;

fn meow<T: ConstParamTy_>() {}

fn main() {}

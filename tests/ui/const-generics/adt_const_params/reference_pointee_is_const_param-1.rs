#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::UnsizedConstParamTy;

struct Foo;

impl UnsizedConstParamTy for &'static Foo {}
//~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type

fn main() {}

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::UnsizedConstParamTy;

trait Trait {}

impl UnsizedConstParamTy for dyn Trait {}
//~^ ERROR: the trait `ConstParamTy` may not be implemented for this type

fn foo<const N: dyn Trait>() {}

fn main() {}

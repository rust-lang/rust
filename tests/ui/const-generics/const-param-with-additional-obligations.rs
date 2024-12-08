#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::UnsizedConstParamTy;

#[derive(Eq, PartialEq)]
struct Foo<T>(T);

trait Other {}

impl<T> UnsizedConstParamTy for Foo<T> where T: Other + UnsizedConstParamTy {}

fn foo<const N: Foo<u8>>() {}
//~^ ERROR `Foo<u8>` must implement `ConstParamTy` to be used as the type of a const generic parameter
//~| NOTE `u8` must implement `Other`, but it does not

fn main() {}

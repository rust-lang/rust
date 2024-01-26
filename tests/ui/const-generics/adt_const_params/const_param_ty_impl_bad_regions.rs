// check that we actually take into account region constraints for `ConstParamTy` impl checks

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

struct Foo<'a>(&'a u32);

struct Bar<T>(T);

impl ConstParamTy for Foo<'static> {}
impl<'a> ConstParamTy for Bar<Foo<'a>> {}
//~^ ERROR the trait `ConstParamTy` cannot be implemented for this type

fn main() {}

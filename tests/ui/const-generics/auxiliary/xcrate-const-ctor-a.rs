#![feature(adt_const_params)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
pub struct Foo;

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(ConstParamTy)]
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this ty
//~| ERROR the trait `ConstParamTy_` cannot be implemented for this ty
struct Foo([*const u8; 1]);

#[derive(ConstParamTy)]
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this ty
//~| ERROR the trait `ConstParamTy_` cannot be implemented for this ty
struct Foo2([*mut u8; 1]);

#[derive(ConstParamTy)]
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this ty
//~| ERROR the trait `ConstParamTy_` cannot be implemented for this ty
struct Foo3([fn(); 1]);

fn main() {}

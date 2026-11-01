#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(ConstParamTy)]
struct Foo([*const u8; 1]);
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this ty

#[derive(ConstParamTy)]
struct Foo2([*mut u8; 1]);
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this ty

#[derive(ConstParamTy)]
struct Foo3([fn(); 1]);
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this ty

fn main() {}

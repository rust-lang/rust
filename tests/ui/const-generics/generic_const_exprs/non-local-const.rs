// regression test for #133808.
//@ aux-build:non_local_type_const.rs

#![feature(min_generic_const_args, macroless_generic_const_args, generic_const_exprs)]
#![allow(incomplete_features)]
#![crate_type = "lib"]
extern crate non_local_type_const;

pub trait Foo {}
impl Foo for [u8; non_local_type_const::NON_LOCAL_CONST] {}
//~^ ERROR the constant `'a'` is not of type `usize`

// regression test for #133808.
//@ aux-build:non_local_type_const.rs

#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
#![crate_type = "lib"]
extern crate non_local_type_const;

pub trait Foo {}
impl Foo for [u8; non_local_type_const::NON_LOCAL_CONST] {}
//~^ ERROR the constant `'a'` is not of type `usize`

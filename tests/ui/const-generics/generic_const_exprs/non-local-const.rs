// regression test for #133808.

#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

pub trait Foo {}
impl Foo for [u8; std::path::MAIN_SEPARATOR] {}
//~^ ERROR the constant `MAIN_SEPARATOR` is not of type `usize`

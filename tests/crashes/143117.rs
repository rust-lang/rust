//@ known-bug: rust-lang/rust#143117
#![feature(generic_const_exprs)]
#![feature(generic_const_parameter_types)]
#![feature(adt_const_params)]
#[derive(Clone)]
struct Foo<const N : [u8; Self::FOO]>;

pub fn main() {}

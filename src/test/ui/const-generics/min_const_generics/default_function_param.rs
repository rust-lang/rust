#![feature(const_generic_defaults)]
#![feature(min_const_generics)]

fn foo<const SIZE: usize = 5>() {}
//~^ ERROR default values for const generic parameters are experimental

fn main() {}

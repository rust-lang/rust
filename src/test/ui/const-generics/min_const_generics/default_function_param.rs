#![crate_type = "lib"]
#![feature(const_generics_defaults)]
#![feature(min_const_generics)]
#![allow(incomplete_features)]

fn foo<const SIZE: usize = 5usize>() {}

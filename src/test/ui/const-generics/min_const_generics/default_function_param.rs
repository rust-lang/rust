#![crate_type = "lib"]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

fn foo<const SIZE: usize = 5usize>() {}
//~^ ERROR defaults for const parameters are

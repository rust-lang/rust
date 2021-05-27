#![crate_type = "lib"]
#![feature(const_generics_defaults)]

fn foo<const SIZE: usize = 5usize>() {}
//~^ ERROR defaults for const parameters are

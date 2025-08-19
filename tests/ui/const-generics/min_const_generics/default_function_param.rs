#![crate_type = "lib"]

fn foo<const SIZE: usize = 5usize>() {}
//~^ ERROR defaults for generic parameters are not allowed here

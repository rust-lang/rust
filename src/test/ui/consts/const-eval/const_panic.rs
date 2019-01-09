#![feature(const_panic)]
#![crate_type = "lib"]

pub const Z: () = panic!("cheese");
//~^ ERROR any use of this value will cause an error

pub const Y: () = unreachable!();
//~^ ERROR any use of this value will cause an error

pub const X: () = unimplemented!();
//~^ ERROR any use of this value will cause an error

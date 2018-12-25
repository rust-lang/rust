#![no_std]
#![crate_type = "lib"]
#![feature(const_panic)]

const Z: () = panic!("cheese");
//~^ ERROR any use of this value will cause an error

const Y: () = unreachable!();
//~^ ERROR any use of this value will cause an error

const X: () = unimplemented!();
//~^ ERROR any use of this value will cause an error

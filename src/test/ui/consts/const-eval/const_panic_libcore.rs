#![no_std]
#![crate_type = "lib"]
#![feature(const_panic)]

const Z: () = panic!("cheese");
//~^ ERROR this constant cannot be used

const Y: () = unreachable!();
//~^ ERROR this constant cannot be used

const X: () = unimplemented!();
//~^ ERROR this constant cannot be used

#![crate_type = "lib"]
#![deny(unused_attributes)]
//~^ NOTE lint level is defined here
#![doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
//~| WARNING will become a hard error in a future release

#[doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
//~| WARNING will become a hard error in a future release
pub fn foo() {}

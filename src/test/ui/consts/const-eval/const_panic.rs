#![feature(const_panic)]
#![crate_type = "lib"]

pub const Z: () = std::panic!("cheese");
//~^ ERROR any use of this value will cause an error

pub const Y: () = std::assert!(1 == 2);
//~^ ERROR any use of this value will cause an error

pub const X: () = std::unimplemented!();
//~^ ERROR any use of this value will cause an error

pub const W: () = core::panic!("cheese");
//~^ ERROR any use of this value will cause an error

pub const V: () = core::assert!(1.2 < 1.0);
//~^ ERROR any use of this value will cause an error

pub const U: () = core::unimplemented!();
//~^ ERROR any use of this value will cause an error

// edition:2021
#![feature(const_panic)]
#![crate_type = "lib"]

const A: () = std::panic!("blåhaj");
//~^ ERROR evaluation of constant value failed

const B: () = std::panic!();
//~^ ERROR evaluation of constant value failed

const C: () = std::unreachable!();
//~^ ERROR evaluation of constant value failed

const D: () = std::unimplemented!();
//~^ ERROR evaluation of constant value failed

const E: () = core::panic!("shark");
//~^ ERROR evaluation of constant value failed

const F: () = core::panic!();
//~^ ERROR evaluation of constant value failed

const G: () = core::unreachable!();
//~^ ERROR evaluation of constant value failed

const H: () = core::unimplemented!();
//~^ ERROR evaluation of constant value failed

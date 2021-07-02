// edition:2021

// NB: panic macros without arguments share the 2015/2018 edition hook
// We cannot annotate the expected error in the test because it point at libcore
// FIXME: this is a very bad error message

#![feature(const_panic)]
#![allow(non_fmt_panic)]
#![crate_type = "lib"]

const Z: () = std::panic!("cheese");

const Z2: () = std::panic!();
//~^ ERROR evaluation of constant value failed

const Y: () = std::unreachable!();
//~^ ERROR evaluation of constant value failed

const X: () = std::unimplemented!();
//~^ ERROR evaluation of constant value failed

const Z_CORE: () = core::panic!("cheese");

const Z2_CORE: () = core::panic!();
//~^ ERROR evaluation of constant value failed

const Y_CORE: () = core::unreachable!();
//~^ ERROR evaluation of constant value failed

const X_CORE: () = core::unimplemented!();
//~^ ERROR evaluation of constant value failed

#![feature(const_panic)]
#![crate_type = "lib"]

const MSG: &str = "hello";

const Z: () = std::panic!("cheese");
//~^ ERROR any use of this value will cause an error

const Z2: () = std::panic!();
//~^ ERROR any use of this value will cause an error

const Y: () = std::unreachable!();
//~^ ERROR any use of this value will cause an error

const X: () = std::unimplemented!();
//~^ ERROR any use of this value will cause an error
//
const W: () = std::panic!(MSG);
//~^ ERROR any use of this value will cause an error

const Z_CORE: () = core::panic!("cheese");
//~^ ERROR any use of this value will cause an error

const Z2_CORE: () = core::panic!();
//~^ ERROR any use of this value will cause an error

const Y_CORE: () = core::unreachable!();
//~^ ERROR any use of this value will cause an error

const X_CORE: () = core::unimplemented!();
//~^ ERROR any use of this value will cause an error

const W_CORE: () = core::panic!(MSG);
//~^ ERROR any use of this value will cause an error

//@ edition:2021
#![crate_type = "lib"]

const MSG: &str = "hello";

const A: () = std::panic!("blåhaj");
//~^ ERROR evaluation panicked

const B: () = std::panic!();
//~^ ERROR evaluation panicked

const C: () = std::unreachable!();
//~^ ERROR evaluation panicked

const D: () = std::unimplemented!();
//~^ ERROR evaluation panicked

const E: () = std::panic!("{}", MSG);
//~^ ERROR evaluation panicked

const A_CORE: () = core::panic!("shark");
//~^ ERROR evaluation panicked

const B_CORE: () = core::panic!();
//~^ ERROR evaluation panicked

const C_CORE: () = core::unreachable!();
//~^ ERROR evaluation panicked

const D_CORE: () = core::unimplemented!();
//~^ ERROR evaluation panicked

const E_CORE: () = core::panic!("{}", MSG);
//~^ ERROR evaluation panicked

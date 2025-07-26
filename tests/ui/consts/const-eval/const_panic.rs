#![allow(non_fmt_panics)]
#![crate_type = "lib"]

const MSG: &str = "hello";

const Z: () = std::panic!("cheese");
//~^ ERROR evaluation panicked

const Z2: () = std::panic!();
//~^ ERROR evaluation panicked

const Y: () = std::unreachable!();
//~^ ERROR evaluation panicked

const X: () = std::unimplemented!();
//~^ ERROR evaluation panicked

const W: () = std::panic!(MSG);
//~^ ERROR evaluation panicked

const W2: () = std::panic!("{}", MSG);
//~^ ERROR evaluation panicked

const Z_CORE: () = core::panic!("cheese");
//~^ ERROR evaluation panicked

const Z2_CORE: () = core::panic!();
//~^ ERROR evaluation panicked

const Y_CORE: () = core::unreachable!();
//~^ ERROR evaluation panicked

const X_CORE: () = core::unimplemented!();
//~^ ERROR evaluation panicked

const W_CORE: () = core::panic!(MSG);
//~^ ERROR evaluation panicked

const W2_CORE: () = core::panic!("{}", MSG);
//~^ ERROR evaluation panicked

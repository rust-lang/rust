//@ edition: 2018

macro_rules! define_other_core {
    ( ) => {
        extern crate std as core;
        //~^ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    };
}

fn main() {
    core::panic!(); //~ ERROR `core` is ambiguous
    ::core::panic!(); //~ ERROR `core` is ambiguous
}

define_other_core!();

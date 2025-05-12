macro_rules! define_other_core {
    ( ) => {
        extern crate std as core;
        //~^ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    };
}

fn main() {
    core::panic!();
}

define_other_core!();

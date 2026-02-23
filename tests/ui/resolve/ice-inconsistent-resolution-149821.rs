//@ edition: 2024

mod m {
    use crate::*;
    use core;
}

macro_rules! define_other_core {
    () => {
        extern crate std as core;
        //~^ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    };
}

define_other_core! {}

fn main() {}

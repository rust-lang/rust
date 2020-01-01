// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
// aux-build:two_macros.rs

macro_rules! define_vec {
    () => {
        extern crate std as Vec;
    }
}

define_vec!();

mod m {
    fn check() {
        Vec::panic!(); //~ ERROR `Vec` is ambiguous
    }
}

macro_rules! define_other_core {
    () => {
        extern crate std as core;
        //~^ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    }
}

define_other_core!();

fn main() {}

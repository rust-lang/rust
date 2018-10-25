// aux-build:two_macros.rs

#![feature(extern_crate_item_prelude)]

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

fn main() {}

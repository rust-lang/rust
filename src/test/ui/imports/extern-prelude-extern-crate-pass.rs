// compile-pass
// aux-build:two_macros.rs

#![feature(extern_crate_item_prelude)]

extern crate two_macros;

mod m {
    fn check() {
        two_macros::m!(); // OK
    }
}

fn main() {}

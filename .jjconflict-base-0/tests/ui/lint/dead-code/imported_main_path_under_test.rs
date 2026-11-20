//@ check-pass
//@ compile-flags: --test

#![deny(dead_code)]

mod m {
    pub fn main() {}
}

use m::main;

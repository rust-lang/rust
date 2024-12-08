//@ run-pass

#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;
extern crate rustc_span;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

mod submod {
    use rustc_macros::{Decodable, Encodable};

    // if any of these are implemented without global calls for any
    // function calls, then being in a submodule will (correctly)
    // cause errors about unrecognised module `std` (or `extra`)
    #[allow(dead_code)]
    #[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone, Debug, Encodable, Decodable)]
    enum A {
        A1(usize),
        A2(isize),
    }

    #[allow(dead_code)]
    #[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone, Debug, Encodable, Decodable)]
    struct B {
        x: usize,
        y: isize,
    }

    #[allow(dead_code)]
    #[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone, Debug, Encodable, Decodable)]
    struct C(usize, isize);
}

pub fn main() {}

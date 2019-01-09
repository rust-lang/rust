#![feature(rustc_private)]

extern crate serialize;
use serialize as rustc_serialize;

mod submod {
    // if any of these are implemented without global calls for any
    // function calls, then being in a submodule will (correctly)
    // cause errors about unrecognised module `std` (or `extra`)
    #[derive(PartialEq, PartialOrd, Eq, Ord,
               Hash,
               Clone,
               Debug,
               RustcEncodable, RustcDecodable)]
    enum A { A1(usize), A2(isize) }

    #[derive(PartialEq, PartialOrd, Eq, Ord,
               Hash,
               Clone,
               Debug,
               RustcEncodable, RustcDecodable)]
    struct B { x: usize, y: isize }

    #[derive(PartialEq, PartialOrd, Eq, Ord,
               Hash,
               Clone,
               Debug,
               RustcEncodable, RustcDecodable)]
    struct C(usize, isize);

}

pub fn main() {}

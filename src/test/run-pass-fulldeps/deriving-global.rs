// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

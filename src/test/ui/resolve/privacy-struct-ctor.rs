// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:privacy-struct-ctor.rs

#![feature(pub_restricted)]

extern crate privacy_struct_ctor as xcrate;

mod m {
    pub struct S(u8);

    pub mod n {
        pub(m) struct Z(pub(m::n) u8);
    }

    use m::n::Z; // OK, only the type is imported

    fn f() {
        n::Z; //~ ERROR tuple struct `Z` is private
        Z;
        //~^ ERROR expected value, found struct `Z`
        //~| NOTE tuple struct constructors with private fields are invisible outside of their mod
    }
}

use m::S; // OK, only the type is imported

fn main() {
    m::S; //~ ERROR tuple struct `S` is private
    S;
    //~^ ERROR expected value, found struct `S`
    //~| NOTE constructor is not visible here due to private fields
    m::n::Z; //~ ERROR tuple struct `Z` is private

    xcrate::m::S; //~ ERROR tuple struct `S` is private
    xcrate::S;
    //~^ ERROR expected value, found struct `xcrate::S`
    //~| NOTE constructor is not visible here due to private fields
    xcrate::m::n::Z; //~ ERROR tuple struct `Z` is private
}

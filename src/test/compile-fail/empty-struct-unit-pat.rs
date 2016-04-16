// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Can't use unit struct as enum pattern

// aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty2;

enum E {
    Empty4
}

// remove attribute after warning cycle and promoting warnings to errors
fn main() {
    let e2 = Empty2;
    let e4 = E::Empty4;
    let xe2 = XEmpty2;
    let xe4 = XE::XEmpty4;

    // Rejected by parser as yet
    // match e2 {
    //     Empty2() => () // ERROR `Empty2` does not name a tuple variant or a tuple struct
    // }
    // match xe2 {
    //     XEmpty2() => () // ERROR `XEmpty2` does not name a tuple variant or a tuple struct
    // }
    match e2 {
        Empty2(..) => () //~ ERROR `Empty2` does not name a tuple variant or a tuple struct
            //~^ WARNING hard error
    }
    match xe2 {
        XEmpty2(..) => () //~ ERROR `XEmpty2` does not name a tuple variant or a tuple struct
            //~^ WARNING hard error
    }
    // Rejected by parser as yet
    // match e4 {
    //     E::Empty4() => () // ERROR `E::Empty4` does not name a tuple variant or a tuple struct
    // }
    // match xe4 {
    //     XE::XEmpty4() => (), // ERROR `XE::XEmpty4` does not name a tuple variant or a tuple
    //     _ => {},
    // }
    match e4 {
        E::Empty4(..) => () //~ ERROR `E::Empty4` does not name a tuple variant or a tuple struct
            //~^ WARNING hard error
    }
    match xe4 {
        XE::XEmpty4(..) => (), //~ ERROR `XE::XEmpty4` does not name a tuple variant or a tuple
            //~^ WARNING hard error
        _ => {},
    }
}

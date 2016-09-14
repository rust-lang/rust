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

#![feature(relaxed_adts)]

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

    match e2 {
        Empty2(..) => () //~ ERROR expected tuple struct/variant, found unit struct `Empty2`
            //~^ WARNING hard error
    }
    match xe2 {
        XEmpty2(..) => () //~ ERROR expected tuple struct/variant, found unit struct `XEmpty2`
            //~^ WARNING hard error
    }

    match e4 {
        E::Empty4(..) => () //~ ERROR expected tuple struct/variant, found unit variant `E::Empty4`
            //~^ WARNING hard error
    }
    match xe4 {
        XE::XEmpty4(..) => (),
            //~^ ERROR expected tuple struct/variant, found unit variant `XE::XEmpty4`
            //~| WARNING hard error
        _ => {},
    }
}

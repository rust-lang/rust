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

struct Empty2();

enum E {
    Empty4()
}

// remove attribute after warning cycle and promoting warnings to errors
fn main() {
    let e2 = Empty2();
    let e4 = E::Empty4();
    let xe6 = XEmpty6();
    let xe5 = XE::XEmpty5();

    match e2 {
        Empty2 => () //~ ERROR match bindings cannot shadow tuple structs
    }
    match xe6 {
        XEmpty6 => () //~ ERROR match bindings cannot shadow tuple structs
    }

    match e4 {
        E::Empty4 => ()
        //~^ ERROR expected unit struct/variant or constant, found tuple variant `E::Empty4`
    }
    match xe5 {
        XE::XEmpty5 => (),
        //~^ ERROR expected unit struct/variant or constant, found tuple variant `XE::XEmpty5`
        _ => {},
    }
}

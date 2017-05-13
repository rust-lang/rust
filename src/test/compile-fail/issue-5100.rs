// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_patterns)]
#![feature(box_syntax)]

enum A { B, C }

fn main() {
    match (true, false) {
        A::B => (),
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `A`
//~| expected tuple, found enum `A`
        _ => ()
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `(_, _, _)`
//~| expected a tuple with 2 elements, found one with 3 elements
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `(_, _, _)`
//~| expected a tuple with 2 elements, found one with 3 elements
    }

    match (true, false) {
        box (true, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `Box<_>`
//~| expected tuple, found box
    }

    match (true, false) {
        &(true, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `&_`
//~| expected tuple, found reference
    }


    let v = [('a', 'b')   //~ ERROR expected function, found `(char, char)`
             ('c', 'd'),
             ('e', 'f')];

    for &(x,y) in &v {} // should be OK

    // Make sure none of the errors above were fatal
    let x: char = true; //~  ERROR mismatched types
                        //~| expected char, found bool
}

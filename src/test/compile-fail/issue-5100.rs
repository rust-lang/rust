// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum A { B, C }

fn main() {
    match (true, false) {
        B => (), //~ ERROR expected `(bool,bool)` but found an enum or structure pattern
        _ => ()
    }

    match (true, false) {
        (true, false, false) => () //~ ERROR mismatched types: expected `(bool,bool)` but found tuple (expected a tuple with 2 elements but found one with 3 elements)
    }

    match (true, false) {
        @(true, false) => () //~ ERROR mismatched types: expected `(bool,bool)` but found an @-box pattern
    }

    match (true, false) {
        ~(true, false) => () //~ ERROR mismatched types: expected `(bool,bool)` but found a ~-box pattern
    }

    match (true, false) {
        &(true, false) => () //~ ERROR mismatched types: expected `(bool,bool)` but found an &-pointer pattern
    }


    let v = [('a', 'b')   //~ ERROR expected function but found `(char,char)`
             ('c', 'd'),
             ('e', 'f')];

    for v.each |&(x,y)| {} // should be OK

    // Make sure none of the errors above were fatal
    let x: char = true; //~ ERROR expected `char` but found `bool`
}

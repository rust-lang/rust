// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

fn main() {
    match "foo".to_string() {
        ['f', 'o', ..] => {}
        //~^ ERROR expected an array or slice, found `std::string::String`
        _ => { }
    };

    match &[0, 1, 2] {
        [..] => {} //~ ERROR expected an array or slice, found `&[{integer}; 3]`
    };

    match &[0, 1, 2] {
        &[..] => {} // ok
    };

    match [0, 1, 2] {
        [0] => {}, //~ ERROR pattern requires

        [0, 1, x..] => {
            let a: [_; 1] = x;
        }
        [0, 1, 2, 3, x..] => {} //~ ERROR pattern requires
    };

    match does_not_exist { //~ ERROR cannot find value `does_not_exist` in this scope
        [] => {}
    };
}

fn another_fn_to_avoid_suppression() {
    match Default::default()
    {
        [] => {}  //~ ERROR the type of this value
    };
}

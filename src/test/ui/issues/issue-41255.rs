// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Matching against float literals should result in a linter error

#![feature(exclusive_range_pattern)]
#![allow(unused)]
#![forbid(illegal_floating_point_literal_pattern)]

fn main() {
    let x = 42.0;
    match x {
        5.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                   //~| WARNING hard error
        5.0f32 => {}, //~ ERROR floating-point types cannot be used in patterns
                      //~| WARNING hard error
        -5.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                    //~| WARNING hard error
        1.0 .. 33.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                           //~| WARNING hard error
                           //~| ERROR floating-point types cannot be used in patterns
                           //~| WARNING hard error
        39.0 ..= 70.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                             //~| WARNING hard error
                             //~| ERROR floating-point types cannot be used in patterns
                             //~| WARNING hard error
        _ => {},
    };
    let y = 5.0;
    // Same for tuples
    match (x, 5) {
        (3.14, 1) => {}, //~ ERROR floating-point types cannot be used
                         //~| WARNING hard error
        _ => {},
    }
    // Or structs
    struct Foo { x: f32 };
    match (Foo { x }) {
        Foo { x: 2.0 } => {}, //~ ERROR floating-point types cannot be used
                              //~| WARNING hard error
        _ => {},
    }
}

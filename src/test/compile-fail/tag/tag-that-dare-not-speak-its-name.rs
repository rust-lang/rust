// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #876

#![no_implicit_prelude]
use std::vec::Vec;

fn last<T>(v: Vec<&T> ) -> std::option::Option<T> {
    panic!();
}

fn main() {
    let y;
    let x : char = last(y);
    //~^ ERROR mismatched types
    //~| expected type `char`
    //~| found type `std::option::Option<_>`
    //~| expected char, found enum `std::option::Option`
}

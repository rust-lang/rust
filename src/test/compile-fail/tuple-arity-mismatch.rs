// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #6155

fn first((value, _): (isize, f64)) -> isize { value }

fn main() {
    let y = first ((1,2.0,3));
    //~^ ERROR mismatched types
    //~| expected type `(isize, f64)`
    //~| found type `(isize, f64, {integer})`
    //~| expected a tuple with 2 elements, found one with 3 elements

    let y = first ((1,));
    //~^ ERROR mismatched types
    //~| expected type `(isize, f64)`
    //~| found type `(isize,)`
    //~| expected a tuple with 2 elements, found one with 1 elements
}

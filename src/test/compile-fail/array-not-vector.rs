// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _x: isize = [1is, 2, 3];
    //~^ ERROR mismatched types
    //~| expected `isize`
    //~| found `[isize; 3]`
    //~| expected isize
    //~| found array of 3 elements

    let x: &[isize] = &[1, 2, 3];
    let _y: &isize = x;
    //~^ ERROR mismatched types
    //~| expected `&isize`
    //~| found `&[isize]`
    //~| expected isize
    //~| found slice
}

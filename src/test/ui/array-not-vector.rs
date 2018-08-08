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
    let _x: i32 = [1, 2, 3];
    //~^ ERROR mismatched types
    //~| expected type `i32`
    //~| found type `[{integer}; 3]`
    //~| expected i32, found array of 3 elements

    let x: &[i32] = &[1, 2, 3];
    let _y: &i32 = x;
    //~^ ERROR mismatched types
    //~| expected type `&i32`
    //~| found type `&[i32]`
    //~| expected i32, found slice
}

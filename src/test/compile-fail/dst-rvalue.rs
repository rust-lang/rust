// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that dynamically sized rvalues are forbidden

pub fn main() {
    let _x: Box<str> = box *"hello world";
    //~^ ERROR E0161

    let array: &[int] = &[1, 2, 3];
    let _x: Box<[int]> = box *array;
    //~^ ERROR E0161
}

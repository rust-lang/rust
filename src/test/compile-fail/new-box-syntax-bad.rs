// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

// Tests that the new `box` syntax works with unique pointers and GC pointers.

use std::gc::{Gc, GC};
use std::owned::{Box, HEAP};

pub fn main() {
    let x: Gc<int> = box(HEAP) 2;  //~ ERROR mismatched types
    let y: Gc<int> = box(HEAP)(1 + 2);  //~ ERROR mismatched types
    let z: Box<int> = box(GC)(4 + 5);   //~ ERROR mismatched types
}


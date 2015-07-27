// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code, unused_variables)]
#![feature(box_heap)]
#![feature(placement_in_syntax)]

// Tests that the new `in` syntax works with unique pointers.
//
// Compare with new-box-syntax.rs

use std::boxed::{Box, HEAP};

struct Structure {
    x: isize,
    y: isize,
}

pub fn main() {
    let x: Box<isize> = in HEAP { 2 };
    let b: Box<isize> = in HEAP { 1 + 2 };
    let c = in HEAP { 3 + 4 };

    let s: Box<Structure> = in HEAP {
        Structure {
            x: 3,
            y: 4,
        }
    };
}

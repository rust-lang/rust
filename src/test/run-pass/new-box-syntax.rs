// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(dead_code, unused_variables)]
#![feature(box_syntax, box_heap)]
#![feature(placement_in_syntax)]

// during check-pretty, the expanded code needs to opt into these
// features
#![feature(placement_new_protocol, core_intrinsics)]

// Tests that the new `box` syntax works with unique pointers.

use std::boxed::{Box, HEAP};

struct Structure {
    x: isize,
    y: isize,
}

pub fn main() {
    let x: Box<isize> = in HEAP { 2 };
    let y: Box<isize> = box 2;
    let b: Box<isize> = box (1 + 2);
    let c = box (3 + 4);

    let s: Box<Structure> = box Structure {
        x: 3,
        y: 4,
    };
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-pretty-expanded FIXME #26067

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(warnings)]
#![feature(box_syntax, box_heap)]
#![feature(placement_in_syntax)]

// Tests that the new `box` syntax works with unique pointers.

use std::boxed::{Box, HEAP};

struct Structure {
    x: isize,
    y: isize,
}

pub fn main() {
    // FIXME (#22181) put back when new placement-in syntax is supported
    // let x: Box<isize> = in HEAP { 2 };
    let y: Box<isize> = box 2;
    let b: Box<isize> = box () (1 + 2);
    let c: Box<_> = box () (3 + 4);
}

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

// Tests that the new `box` syntax works with unique pointers.

use std::boxed::{Box, HEAP};

struct Structure {
    x: int,
    y: int,
}

pub fn main() {
    let x: Box<int> = box(HEAP) 2i;
    let y: Box<int> = box 2i;
    let b: Box<int> = box()(1i + 2);
    let c = box()(3i + 4);
}

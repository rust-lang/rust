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

use std::gc::Gc;
use std::owned::HEAP;

struct Structure {
    x: int,
    y: int,
}

pub fn main() {
    let x: ~int = box(HEAP) 2;
    let y: ~int = box 2;
    let z: Gc<int> = box(GC) 2;
    let a: Gc<Structure> = box(GC) Structure {
        x: 10,
        y: 20,
    };
    let b: ~int = box()(1 + 2);
    let c = box()(3 + 4);
    let d = box(GC)(5 + 6);
}


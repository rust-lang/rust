// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_assignment)]

use std::gc::{Gc, GC};

pub fn main() {
    let x : [Gc<int>, ..5] = [box(GC) 1,box(GC) 2,box(GC) 3,box(GC) 4,box(GC) 5];
    let _y : [Gc<int>, ..5] = [box(GC) 1,box(GC) 2,box(GC) 3,box(GC) 4,box(GC) 5];
    let mut z = [box(GC) 1,box(GC) 2,box(GC) 3,box(GC) 4,box(GC) 5];
    z = x;
    assert_eq!(*z[0], 1);
    assert_eq!(*z[4], 5);
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::task;
use std::gc::{GC, Gc};

fn adder(x: Gc<int>, y: Gc<int>) -> int { return *x + *y; }
fn failer() -> Gc<int> { fail!(); }
pub fn main() {
    assert!(task::try(proc() {
        adder(box(GC) 2, failer()); ()
    }).is_err());
}

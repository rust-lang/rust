// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:fail

#![feature(managed_boxes)]

use std::gc::GC;

fn f() -> Vec<int> { fail!(); }

// Voodoo. In unwind-alt we had to do this to trigger the bug. Might
// have been to do with memory allocation patterns.
fn prime() {
    box(GC) 0i;
}

fn partial() {
    let _x = box f();
}

fn main() {
    prime();
    partial();
}

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

fn user(_i: int) {}

fn foo() {
    // Here, i is *copied* into the proc (heap closure).
    // Requires allocation.  The proc's copy is not mutable.
    let mut i = 0;
    do task::spawn {
        user(i);
        println!("spawned {}", i)
    }
    i += 1;
    println!("original {}", i)
}

fn bar() {
    // Here, the original i has not been moved, only copied, so is still
    // mutable outside of the proc.
    let mut i = 0;
    while i < 10 {
        do task::spawn {
            user(i);
        }
        i += 1;
    }
}

fn car() {
    // Here, i must be shadowed in the proc to be mutable.
    let mut i = 0;
    while i < 10 {
        do task::spawn {
            let mut i = i;
            i += 1;
            user(i);
        }
        i += 1;
    }
}

pub fn main() {}


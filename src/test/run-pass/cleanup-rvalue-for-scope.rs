// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the lifetime of rvalues in for loops is extended
// to the for loop itself.

#![feature(macro_rules)]

use std::ops::Drop;

static mut FLAGS: u64 = 0;

struct Box<T> { f: T }
struct AddFlags { bits: u64 }

fn AddFlags(bits: u64) -> AddFlags {
    AddFlags { bits: bits }
}

fn arg(exp: u64, _x: &AddFlags) {
    check_flags(exp);
}

fn pass<T>(v: T) -> T {
    v
}

fn check_flags(exp: u64) {
    unsafe {
        let x = FLAGS;
        FLAGS = 0;
        println!("flags {}, expected {}", x, exp);
        assert_eq!(x, exp);
    }
}

impl AddFlags {
    fn check_flags<'a>(&'a self, exp: u64) -> &'a AddFlags {
        check_flags(exp);
        self
    }

    fn bits(&self) -> u64 {
        self.bits
    }
}

impl Drop for AddFlags {
    fn drop(&mut self) {
        unsafe {
            FLAGS = FLAGS + self.bits;
        }
    }
}

pub fn main() {
    // The array containing [AddFlags] should not be dropped until
    // after the for loop:
    for x in [AddFlags(1)].iter() {
        check_flags(0);
    }
    check_flags(1);
}

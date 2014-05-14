// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate testcrate;

use std::mem;

#[link(name = "test", kind = "static")]
extern {
    fn give_back(tu: testcrate::TestUnion) -> u64;
}

fn main() {
    let magic: u64 = 0xDEADBEEF;

    // Let's test calling it cross crate
    let back = unsafe {
        testcrate::give_back(mem::transmute(magic))
    };
    assert_eq!(magic, back);

    // And just within this crate
    let back = unsafe {
        give_back(mem::transmute(magic))
    };
    assert_eq!(magic, back);
}

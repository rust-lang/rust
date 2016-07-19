// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static global: isize = 3;

static global0: isize = 4;

pub static global2: &'static isize = &global0;

pub fn verify_same(a: &'static isize) {
    let a = a as *const isize as usize;
    let b = &global as *const isize as usize;
    assert_eq!(a, b);
}

pub fn verify_same2(a: &'static isize) {
    let a = a as *const isize as usize;
    let b = global2 as *const isize as usize;
    assert_eq!(a, b);
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test coercions involving DST and/or raw pointers

struct S;
trait T {}
impl T for S {}

pub fn main() {
    let x: &T = &S;
    // Test we can convert from &-ptr to *-ptr of trait objects
    let x: *const T = &S;

    // Test we can convert from &-ptr to *-ptr of struct pointer (not DST)
    let x: *const S = &S;

    // As above, but mut
    let x: &mut T = &mut S;
    let x: *mut T = &mut S;

    let x: *mut S = &mut S;

    // Test we can chnage the mutability from mut to const.
    let x: &T = &mut S;
    let x: *const T = &mut S;
}

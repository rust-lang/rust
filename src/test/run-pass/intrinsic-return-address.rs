// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(intrinsics)]

use std::ptr;

struct Point {
    x: f32,
    y: f32,
    z: f32,
}

extern "rust-intrinsic" {
    fn return_address() -> *const u8;
}

fn f(result: &mut uint) -> Point {
    unsafe {
        *result = return_address() as uint;
        Point {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        }
    }

}

fn main() {
    let mut intrinsic_reported_address = 0;
    let pt = f(&mut intrinsic_reported_address);
    let actual_address = &pt as *const Point as uint;
    assert_eq!(intrinsic_reported_address, actual_address);
}


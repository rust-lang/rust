// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::simd::{i32x4, f32x4};

fn test_int(e: i32) -> i32 {
    let v = i32x4(e, 0i32, 0i32, 0i32);
    let i32x4(e2, _, _, _) = v * v + v - v;
    e2
}

fn test_float(e: f32) -> f32 {
    let v = f32x4(e, 0f32, 0f32, 0f32);
    let f32x4(e2, _, _, _) = v * v + v - v;
    e2
}

fn main() {
    assert_eq!(test_int(3i32), 9i32);
    assert_eq!(test_float(3f32), 9f32);
}

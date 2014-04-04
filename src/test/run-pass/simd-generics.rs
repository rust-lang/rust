// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

#![feature(simd)]

use std::ops;

#[simd] struct f32x4(f32, f32, f32, f32);

fn add<T: ops::Add<T, T>>(lhs: T, rhs: T) -> T {
    lhs + rhs
}

impl ops::Add<f32x4, f32x4> for f32x4 {
    fn add(&self, rhs: &f32x4) -> f32x4 {
        *self + *rhs
    }
}

pub fn main() {
    let lr = f32x4(1.0f32, 2.0f32, 3.0f32, 4.0f32);

    // lame-o
    let f32x4(x, y, z, w) = add(lr, lr);
    assert_eq!(x, 2.0f32);
    assert_eq!(y, 4.0f32);
    assert_eq!(z, 6.0f32);
    assert_eq!(w, 8.0f32);
}

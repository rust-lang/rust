// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// checks that struct simd types can be swizzled.

#[feature(simd, phase)];
#[allow(experimental)];
#[phase(syntax)] extern crate simd_syntax;
extern crate simd;

use simd::f32x4;

#[simd]
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

#[inline(never)]
fn get(v: RGBA) -> f32x4 {
    swizzle_simd!(v -> (3, 2, 1, 0))
}


pub fn main() {
    let v1 = RGBA {
        r: 1.0f32,
        g: 2.0f32,
        b: 3.0f32,
        a: 4.0f32,
    };
    let v2 = get(v1);
    assert_eq!(v1.r, v2[3]);
    assert_eq!(v1.g, v2[2]);
    assert_eq!(v1.b, v2[1]);
    assert_eq!(v1.a, v2[0]);
}

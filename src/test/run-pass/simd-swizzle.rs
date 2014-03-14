// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// xfail-stage1

#[feature(simd, phase)];
#[allow(experimental)];

#[phase(syntax)]
extern crate simd_syntax;
extern crate simd;

use simd::BoolSimd;
use simd::f32x4;

#[simd]
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

static const_expr: i32 = 15 / 3;

fn main() {
    let v = gather_simd!(1.0, 2.0, 4.0, 8.0);
    let rev_v = swizzle_simd!(v -> (3, 2, 1, 0));
    let gather_rev = gather_simd!(8.0, 4.0, 2.0, 1.0);
    let cond = rev_v == gather_rev;
    assert!(cond.every_true());
    assert!((swizzle_simd!(v -> (0, 1, 2, 3,
                                 3, 2, 1, 0)) ==
             gather_simd!(1.0, 2.0, 4.0, 8.0,
                          8.0, 4.0, 2.0, 1.0)).every_true());
    assert_eq!(swizzle_simd!(v -> (0))[0], 1.0);

    assert_eq!(swizzle_simd!(v -> (const_expr - 2))[0], 8.0);

    let rgba = RGBA{ r: 1.0f32,
                     g: 2.0f32,
                     b: 3.0f32,
                     a: 4.0f32 };

    assert!((swizzle_simd!(rgba -> (3, 2, 1, 0)) ==
             gather_simd!(4.0, 3.0, 2.0, 1.0)).every_true());
}

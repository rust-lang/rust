// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(simd, phase)];

#[phase(syntax)]
extern crate simd_syntax;
extern crate simd;
use simd::{f64x8, i64x4};

#[inline(never)] fn i64x4_f(_: i64x4) {}
#[inline(never)] fn f64x8_f(_: f64x8) {}

pub fn main() {
    let v = gather_simd!(0, 1, 2, 3);
    i64x4_f(v);

    let v = gather_simd!(1.0, 2.0, 3.0, 4.0);
    let v = swizzle_simd!(v -> (3, 2, 1, 0, 0, 1, 2, 3));
    f64x8_f(v);
}

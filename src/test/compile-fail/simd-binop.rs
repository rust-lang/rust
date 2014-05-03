// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![allow(experimental)]

use std::unstable::simd::f32x4;

fn main() {

    let _ = f32x4(0.0, 0.0, 0.0, 0.0) == f32x4(0.0, 0.0, 0.0, 0.0);
    //~^ ERROR binary comparison operation `==` not supported for floating point SIMD vector `std::unstable::simd::f32x4`

    let _ = f32x4(0.0, 0.0, 0.0, 0.0) != f32x4(0.0, 0.0, 0.0, 0.0);
    //~^ ERROR binary comparison operation `!=` not supported for floating point SIMD vector `std::unstable::simd::f32x4`

    let _ = f32x4(0.0, 0.0, 0.0, 0.0) < f32x4(0.0, 0.0, 0.0, 0.0);
    //~^ ERROR binary comparison operation `<` not supported for floating point SIMD vector `std::unstable::simd::f32x4`

    let _ = f32x4(0.0, 0.0, 0.0, 0.0) <= f32x4(0.0, 0.0, 0.0, 0.0);
    //~^ ERROR binary comparison operation `<=` not supported for floating point SIMD vector `std::unstable::simd::f32x4`

    let _ = f32x4(0.0, 0.0, 0.0, 0.0) >= f32x4(0.0, 0.0, 0.0, 0.0);
    //~^ ERROR binary comparison operation `>=` not supported for floating point SIMD vector `std::unstable::simd::f32x4`

    let _ = f32x4(0.0, 0.0, 0.0, 0.0) > f32x4(0.0, 0.0, 0.0, 0.0);
    //~^ ERROR binary comparison operation `>` not supported for floating point SIMD vector `std::unstable::simd::f32x4`

}

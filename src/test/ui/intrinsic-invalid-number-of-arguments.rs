// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test number of arguments in platform-specific intrinsic function
// This is the error E0444

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd(x: f64x2, y: f64x2, z: f64x2) -> i32; //~ platform-specific intrinsic
}

pub fn main() {
}

// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten
// min-llvm-version 6.0

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

extern "platform-intrinsic" {
    fn simd_fmin<T>(x: T, y: T) -> T;
    fn simd_fmax<T>(x: T, y: T) -> T;
}

// CHECK-LABEL: @fmin
#[no_mangle]
pub unsafe fn fmin(a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: call <4 x float> @llvm.minnum.v4f32
    simd_fmin(a, b)
}

// FIXME(49261)
// // C_HECK-LABEL: @fmax
// #[no_mangle]
// pub unsafe fn fmax(a: f32x4, b: f32x4) -> f32x4 {
// // C_HECK: call <4 x float> @llvm.maxnum.v4f32
//     simd_fmax(a, b)
// }

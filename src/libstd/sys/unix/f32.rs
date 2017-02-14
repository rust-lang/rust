// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use intrinsics;

pub mod cmath {
    use libc::{c_float, c_int};

    #[link_name = "m"]
    extern {
        pub fn lgammaf_r(n: c_float, sign: &mut c_int) -> c_float;
        pub fn hypotf(x: c_float, y: c_float) -> c_float;
        pub fn acosf(n: c_float) -> c_float;
        pub fn asinf(n: c_float) -> c_float;
        pub fn atan2f(a: c_float, b: c_float) -> c_float;
        pub fn atanf(n: c_float) -> c_float;
        pub fn coshf(n: c_float) -> c_float;
        pub fn frexpf(n: c_float, value: &mut c_int) -> c_float;
        pub fn ldexpf(x: c_float, n: c_int) -> c_float;
        pub fn sinhf(n: c_float) -> c_float;
        pub fn tanf(n: c_float) -> c_float;
        pub fn tanhf(n: c_float) -> c_float;
    }
}

#[inline]
pub fn floor(x: f32) -> f32 {
    unsafe { intrinsics::floorf32(x) }
}

#[inline]
pub fn ceil(x: f32) -> f32 {
    unsafe { intrinsics::ceilf32(x) }
}

#[inline]
pub fn powf(x: f32, n: f32) -> f32 {
    unsafe { intrinsics::powf32(x, n) }
}

#[inline]
pub fn exp(x: f32) -> f32 {
    unsafe { intrinsics::expf32(x) }
}

#[inline]
pub fn ln(x: f32) -> f32 {
    unsafe { intrinsics::logf32(x) }
}

#[inline]
pub fn log2(x: f32) -> f32 {
    #[cfg(target_os = "android")]
    return ::sys::android::log2f32(x);
    #[cfg(not(target_os = "android"))]
    return unsafe { intrinsics::log2f32(x) };
}

#[inline]
pub fn log10(x: f32) -> f32 {
    unsafe { intrinsics::log10f32(x) }
}

#[inline]
pub fn sin(x: f32) -> f32 {
    unsafe { intrinsics::sinf32(x) }
}

#[inline]
pub fn cos(x: f32) -> f32 {
    unsafe { intrinsics::cosf32(x) }
}

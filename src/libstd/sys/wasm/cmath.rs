// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[inline]
pub unsafe fn cbrtf(n: f32) -> f32 {
    f64::cbrt(n as f64) as f32
}

#[inline]
pub unsafe fn expm1f(n: f32) -> f32 {
    f64::exp_m1(n as f64) as f32
}

#[inline]
#[allow(deprecated)]
pub unsafe fn fdimf(a: f32, b: f32) -> f32 {
    f64::abs_sub(a as f64, b as f64) as f32
}

#[inline]
pub unsafe fn log1pf(n: f32) -> f32 {
    f64::ln_1p(n as f64) as f32
}

#[inline]
pub unsafe fn hypotf(x: f32, y: f32) -> f32 {
    f64::hypot(x as f64, y as f64) as f32
}

#[inline]
pub unsafe fn acosf(n: f32) -> f32 {
    f64::acos(n as f64) as f32
}

#[inline]
pub unsafe fn asinf(n: f32) -> f32 {
    f64::asin(n as f64) as f32
}

#[inline]
pub unsafe fn atan2f(n: f32, b: f32) -> f32 {
    f64::atan2(n as f64, b as f64) as f32
}

#[inline]
pub unsafe fn atanf(n: f32) -> f32 {
    f64::atan(n as f64) as f32
}

#[inline]
pub unsafe fn coshf(n: f32) -> f32 {
    f64::cosh(n as f64) as f32
}

#[inline]
pub unsafe fn sinhf(n: f32) -> f32 {
    f64::sinh(n as f64) as f32
}

#[inline]
pub unsafe fn tanf(n: f32) -> f32 {
    f64::tan(n as f64) as f32
}

#[inline]
pub unsafe fn tanhf(n: f32) -> f32 {
    f64::tanh(n as f64) as f32
}

// These symbols are all defined in `compiler-builtins`
extern {
    pub fn acos(n: f64) -> f64;
    pub fn asin(n: f64) -> f64;
    pub fn atan(n: f64) -> f64;
    pub fn atan2(a: f64, b: f64) -> f64;
    pub fn cbrt(n: f64) -> f64;
    pub fn cosh(n: f64) -> f64;
    pub fn expm1(n: f64) -> f64;
    pub fn fdim(a: f64, b: f64) -> f64;
    pub fn log1p(n: f64) -> f64;
    pub fn sinh(n: f64) -> f64;
    pub fn tan(n: f64) -> f64;
    pub fn tanh(n: f64) -> f64;
    pub fn hypot(x: f64, y: f64) -> f64;
}

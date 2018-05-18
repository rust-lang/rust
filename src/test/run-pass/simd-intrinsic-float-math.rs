// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten
// ignore-android

// FIXME: this test fails on arm-android because the NDK version 14 is too old.
// It needs at least version 18. We disable it on all android build bots because
// there is no way in compile-test to disable it for an (arch,os) pair.

// Test that the simd floating-point math intrinsics produce correct results.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct f32x4(pub f32, pub f32, pub f32, pub f32);

extern "platform-intrinsic" {
    fn simd_fsqrt<T>(x: T) -> T;
    fn simd_fabs<T>(x: T) -> T;
    fn simd_fsin<T>(x: T) -> T;
    fn simd_fcos<T>(x: T) -> T;
    fn simd_ceil<T>(x: T) -> T;
    fn simd_fexp<T>(x: T) -> T;
    fn simd_fexp2<T>(x: T) -> T;
    fn simd_floor<T>(x: T) -> T;
    fn simd_fma<T>(x: T, y: T, z: T) -> T;
    fn simd_flog<T>(x: T) -> T;
    fn simd_flog10<T>(x: T) -> T;
    fn simd_flog2<T>(x: T) -> T;
    fn simd_fpow<T>(x: T, y: T) -> T;
    fn simd_fpowi<T>(x: T, y: i32) -> T;
}

fn main() {
    let x = f32x4(1.0, 1.0, 1.0, 1.0);
    let y = f32x4(-1.0, -1.0, -1.0, -1.0);
    let z = f32x4(0.0, 0.0, 0.0, 0.0);

    let h = f32x4(0.5, 0.5, 0.5, 0.5);

    unsafe {
        let r = simd_fabs(y);
        assert_eq!(x, r);

        let r = simd_fcos(z);
        assert_eq!(x, r);

        let r = simd_ceil(h);
        assert_eq!(x, r);

        let r = simd_fexp(z);
        assert_eq!(x, r);

        let r = simd_fexp2(z);
        assert_eq!(x, r);

        let r = simd_floor(h);
        assert_eq!(z, r);

        let r = simd_fma(x, h, h);
        assert_eq!(x, r);

        let r = simd_fsqrt(x);
        assert_eq!(x, r);

        let r = simd_flog(x);
        assert_eq!(z, r);

        let r = simd_flog2(x);
        assert_eq!(z, r);

        let r = simd_flog10(x);
        assert_eq!(z, r);

        let r = simd_fpow(h, x);
        assert_eq!(h, r);

        let r = simd_fpowi(h, 1);
        assert_eq!(h, r);

        let r = simd_fsin(z);
        assert_eq!(z, r);
    }
}

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

// Test that the simd_select intrinsics produces correct results.

#![feature(repr_simd, platform_intrinsics)]
#[allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct b8x4(pub i8, pub i8, pub i8, pub i8);

extern "platform-intrinsic" {
    fn simd_select<T, U>(x: T, a: U, b: U) -> U;
}

fn main() {
    let m0 = b8x4(!0, !0, !0, !0);
    let m1 = b8x4(0, 0, 0, 0);
    let m2 = b8x4(!0, !0, 0, 0);
    let m3 = b8x4(0, 0, !0, !0);
    let m4 = b8x4(!0, 0, !0, 0);

    unsafe {
        let a = i32x4(1, -2, 3, 4);
        let b = i32x4(5, 6, -7, 8);

        let r: i32x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m2, a, b);
        let e = i32x4(1, -2, -7, 8);
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m3, a, b);
        let e = i32x4(5, 6, 3, 4);
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m4, a, b);
        let e = i32x4(1, 6, 3, 8);
        assert_eq!(r, e);
    }

    unsafe {
        let a = u32x4(1, 2, 3, 4);
        let b = u32x4(5, 6, 7, 8);

        let r: u32x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m2, a, b);
        let e = u32x4(1, 2, 7, 8);
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m3, a, b);
        let e = u32x4(5, 6, 3, 4);
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m4, a, b);
        let e = u32x4(1, 6, 3, 8);
        assert_eq!(r, e);
    }

    unsafe {
        let a = f32x4(1., 2., 3., 4.);
        let b = f32x4(5., 6., 7., 8.);

        let r: f32x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m2, a, b);
        let e = f32x4(1., 2., 7., 8.);
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m3, a, b);
        let e = f32x4(5., 6., 3., 4.);
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m4, a, b);
        let e = f32x4(1., 6., 3., 8.);
        assert_eq!(r, e);
    }

    unsafe {
        let t = !0 as i8;
        let f = 0 as i8;
        let a = b8x4(t, f, t, f);
        let b = b8x4(f, f, f, t);

        let r: b8x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m2, a, b);
        let e = b8x4(t, f, f, t);
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m3, a, b);
        let e = b8x4(f, f, t, f);
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m4, a, b);
        let e = b8x4(t, f, t, t);
        assert_eq!(r, e);
    }
}

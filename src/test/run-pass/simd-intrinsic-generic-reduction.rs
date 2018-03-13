// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the simd_reduce_{op} intrinsics produce the correct results.

#![feature(repr_simd, platform_intrinsics)]
#[allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct b8x4(pub i8, pub i8, pub i8, pub i8);

#[repr(simd)]
#[derive(Copy, Clone)]
struct b8x16(
    pub i8, pub i8, pub i8, pub i8,
    pub i8, pub i8, pub i8, pub i8,
    pub i8, pub i8, pub i8, pub i8,
    pub i8, pub i8, pub i8, pub i8
);

extern "platform-intrinsic" {
    fn simd_reduce_add<T, U>(x: T) -> U;
    fn simd_reduce_mul<T, U>(x: T) -> U;
    fn simd_reduce_min<T, U>(x: T) -> U;
    fn simd_reduce_max<T, U>(x: T) -> U;
    fn simd_reduce_and<T, U>(x: T) -> U;
    fn simd_reduce_or<T, U>(x: T) -> U;
    fn simd_reduce_xor<T, U>(x: T) -> U;
    fn simd_reduce_all<T>(x: T) -> bool;
    fn simd_reduce_any<T>(x: T) -> bool;
}

fn main() {
    unsafe {
        let x = i32x4(1, -2, 3, 4);
        let r: i32 = simd_reduce_add(x);
        assert!(r == 6_i32);
        let r: i32 = simd_reduce_mul(x);
        assert!(r == -24_i32);
        let r: i32 = simd_reduce_min(x);
        assert!(r == -21_i32);
        let r: i32 = simd_reduce_max(x);
        assert!(r == 4_i32);

        let x = i32x4(-1, -1, -1, -1);
        let r: i32 = simd_reduce_and(x);
        assert!(r == -1_i32);
        let r: i32 = simd_reduce_or(x);
        assert!(r == -1_i32);
        let r: i32 = simd_reduce_xor(x);
        assert!(r == 0_i32);

        let x = i32x4(-1, -1, 0, -1);
        let r: i32 = simd_reduce_and(x);
        assert!(r == 0_i32);
        let r: i32 = simd_reduce_or(x);
        assert!(r == -1_i32);
        let r: i32 = simd_reduce_xor(x);
        assert!(r == -1_i32);
    }

    unsafe {
        let x = u32x4(1, 2, 3, 4);
        let r: u32 = simd_reduce_add(x);
        assert!(r == 10_u32);
        let r: u32 = simd_reduce_mul(x);
        assert!(r == 24_u32);
        let r: u32 = simd_reduce_min(x);
        assert!(r == 1_u32);
        let r: u32 = simd_reduce_max(x);
        assert!(r == 4_u32);

        let t = u32::max_value();
        let x = u32x4(t, t, t, t);
        let r: u32 = simd_reduce_and(x);
        assert!(r == t);
        let r: u32 = simd_reduce_or(x);
        assert!(r == t);
        let r: u32 = simd_reduce_xor(x);
        assert!(r == 0_u32);

        let x = u32x4(t, t, 0, t);
        let r: u32 = simd_reduce_and(x);
        assert!(r == 0_u32);
        let r: u32 = simd_reduce_or(x);
        assert!(r == t);
        let r: u32 = simd_reduce_xor(x);
        assert!(r == t);
    }

    unsafe {
        let x = f32x4(1., -2., 3., 4.);
        let r: f32 = simd_reduce_add(x);
        assert!(r == 6_f32);
        let r: f32 = simd_reduce_mul(x);
        assert!(r == -24_f32);
        let r: f32 = simd_reduce_min(x);
        assert!(r == -2_f32);
        let r: f32 = simd_reduce_max(x);
        assert!(r == 4_f32);
    }

    unsafe {
        let x = b8x4(!0, !0, !0, !0);
        let r: bool = simd_reduce_all(x);
        //let r: bool = foobar(x);
        assert!(r);
        let r: bool = simd_reduce_any(x);
        assert!(r);

        let x = b8x4(!0, !0, 0, !0);
        let r: bool = simd_reduce_all(x);
        assert!(!r);
        let r: bool = simd_reduce_any(x);
        assert!(r);

        let x = b8x4(0, 0, 0, 0);
        let r: bool = simd_reduce_all(x);
        assert!(!r);
        let r: bool = simd_reduce_any(x);
        assert!(!r);
    }
}

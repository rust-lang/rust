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
// min-llvm-version 6.0

// Test that the simd_{gather,scatter} intrinsics produce the correct results.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct x4<T>(pub T, pub T, pub T, pub T);

extern "platform-intrinsic" {
    fn simd_gather<T, U, V>(x: T, y: U, z: V) -> T;
    fn simd_scatter<T, U, V>(x: T, y: U, z: V) -> ();
}

fn main() {
    let mut x = [0_f32, 1., 2., 3., 4., 5., 6., 7.];

    let default = x4(-3_f32, -3., -3., -3.);
    let s_strided = x4(0_f32, 2., -3., 6.);
    let mask = x4(-1_i32, -1, 0, -1);

    // reading from *const
    unsafe {
        let pointer = &x[0] as *const f32;
        let pointers =  x4(
            pointer.offset(0) as *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // reading from *mut
    unsafe {
        let pointer = &mut x[0] as *mut f32;
        let pointers = x4(
            pointer.offset(0) as *mut f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = &mut x[0] as *mut f32;
        let pointers = x4(
            pointer.offset(0) as *mut f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let values = x4(42_f32, 43_f32, 44_f32, 45_f32);
        simd_scatter(values, pointers, mask);

        assert_eq!(x, [42., 1., 43., 3., 4., 5., 45., 7.]);
    }
}

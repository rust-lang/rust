//@ run-pass
//@ ignore-emscripten

// Test that the simd_{gather,scatter} intrinsics produce the correct results.

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_gather, simd_scatter};

type x4<T> = Simd<T, 4>;

fn main() {
    let mut x = [0_f32, 1., 2., 3., 4., 5., 6., 7.];

    let default = x4::from_array([-3_f32, -3., -3., -3.]);
    let s_strided = x4::from_array([0_f32, 2., -3., 6.]);
    let mask = x4::from_array([-1_i32, -1, 0, -1]);

    // reading from *const
    unsafe {
        let pointer = x.as_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i * 2)));

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // reading from *mut
    unsafe {
        let pointer = x.as_mut_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i * 2)));

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = x.as_mut_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i * 2)));

        let values = x4::from_array([42_f32, 43_f32, 44_f32, 45_f32]);
        simd_scatter(values, pointers, mask);

        assert_eq!(x, [42., 1., 43., 3., 4., 5., 45., 7.]);
    }

    // test modifying array of *const f32
    let mut y = [
        &x[0] as *const f32,
        &x[1] as *const f32,
        &x[2] as *const f32,
        &x[3] as *const f32,
        &x[4] as *const f32,
        &x[5] as *const f32,
        &x[6] as *const f32,
        &x[7] as *const f32,
    ];

    let default = x4::from_array([y[0], y[0], y[0], y[0]]);
    let s_strided = x4::from_array([y[0], y[2], y[0], y[6]]);

    // reading from *const
    unsafe {
        let pointer = y.as_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i * 2)));

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // reading from *mut
    unsafe {
        let pointer = y.as_mut_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i * 2)));

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = y.as_mut_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i * 2)));

        let values = x4::from_array([y[7], y[6], y[5], y[1]]);
        simd_scatter(values, pointers, mask);

        let s = [
            &x[7] as *const f32,
            &x[1] as *const f32,
            &x[6] as *const f32,
            &x[3] as *const f32,
            &x[4] as *const f32,
            &x[5] as *const f32,
            &x[1] as *const f32,
            &x[7] as *const f32,
        ];
        assert_eq!(y, s);
    }
}

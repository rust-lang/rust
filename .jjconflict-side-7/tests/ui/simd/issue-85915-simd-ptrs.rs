//@ run-pass
//@ ignore-emscripten

// Short form of the generic gather/scatter tests,
// verifying simd([*const T; N]) and simd([*mut T; N]) pass typeck and work.
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_gather, simd_scatter};

type cptrx4<T> = Simd<*const T, 4>;

type mptrx4<T> = Simd<*mut T, 4>;

fn main() {
    let mut x = [0_f32, 1., 2., 3., 4., 5., 6., 7.];

    let default = f32x4::from_array([-3_f32, -3., -3., -3.]);
    let s_strided = f32x4::from_array([0_f32, 2., -3., 6.]);
    let mask = i32x4::from_array([-1_i32, -1, 0, -1]);

    // reading from *const
    unsafe {
        let pointer = &x as *const f32;
        let pointers = cptrx4::from_array([
            pointer.offset(0) as *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6),
        ]);

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = &mut x as *mut f32;
        let pointers = mptrx4::from_array([
            pointer.offset(0) as *mut f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6),
        ]);

        let values = f32x4::from_array([42_f32, 43_f32, 44_f32, 45_f32]);
        simd_scatter(values, pointers, mask);

        assert_eq!(x, [42., 1., 43., 3., 4., 5., 45., 7.]);
    }
}

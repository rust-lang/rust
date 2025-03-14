//@ build-fail
//@ ignore-emscripten

// Test that the simd_{gather,scatter} intrinsics produce ok-ish error
// messages when misused.

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::{simd_gather, simd_scatter};

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct x4<T>(pub [T; 4]);

fn main() {
    let mut x = [0_f32, 1., 2., 3., 4., 5., 6., 7.];

    let default = x4([-3_f32, -3., -3., -3.]);
    let s_strided = x4([0_f32, 2., -3., 6.]);

    let mask = x4([-1_i32, -1, 0, -1]);
    let umask = x4([0u16; 4]);
    let fmask = x4([0_f32; 4]);

    let pointer = x.as_mut_ptr();
    let pointers =
        unsafe { x4([pointer.offset(0), pointer.offset(2), pointer.offset(4), pointer.offset(6)]) };

    unsafe {
        simd_gather(default, mask, mask);
        //~^ ERROR expected element type `i32` of second argument `x4<i32>` to be a pointer to the element type `f32`

        simd_gather(default, pointers, umask);
        //~^ ERROR expected element type `u16` of third argument `x4<u16>` to be a signed integer type

        simd_gather(default, pointers, fmask);
        //~^ ERROR expected element type `f32` of third argument `x4<f32>` to be a signed integer type
    }

    unsafe {
        let values = x4([42_f32, 43_f32, 44_f32, 45_f32]);
        simd_scatter(values, mask, mask);
        //~^ ERROR expected element type `i32` of second argument `x4<i32>` to be a pointer to the element type `f32` of the first argument `x4<f32>`, found `i32` != `*mut f32`

        simd_scatter(values, pointers, umask);
        //~^ ERROR expected element type `u16` of third argument `x4<u16>` to be a signed integer type

        simd_scatter(values, pointers, fmask);
        //~^ ERROR expected element type `f32` of third argument `x4<f32>` to be a signed integer type
    }
}

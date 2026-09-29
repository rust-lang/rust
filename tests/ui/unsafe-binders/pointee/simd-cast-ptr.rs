//@ compile-flags: -Znext-solver
//@ build-pass

// `simd_cast_ptr` should accept vectors of thin pointers to unsafe binders.

#![feature(unsafe_binders, repr_simd, core_intrinsics)]
#![allow(incomplete_features)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_cast_ptr;

fn main() {
    let x = 1u8;
    let ptrs: Simd<*const u8, 2> = Simd([&x as *const u8, std::ptr::null()]);
    let cast: Simd<*const unsafe<> u8, 2> = unsafe { simd_cast_ptr(ptrs) };
    assert!(cast.into_array()[0] as *const u8 == &x as *const u8);
}

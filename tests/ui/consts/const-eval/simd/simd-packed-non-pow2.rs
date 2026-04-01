//@ check-pass
// Fixes #151537
#![feature(portable_simd, core_intrinsics)]
use std::intrinsics::simd::SimdAlign;
use std::{ptr::null, simd::prelude::*};

const _: () = {
    let c = Simd::from_array([0; 3]);
    unsafe {
        core::intrinsics::simd::simd_masked_store::<_, _, _, { SimdAlign::Element }>(
            c,
            null::<i32>(),
            c,
        )
    };
};

fn main() {}

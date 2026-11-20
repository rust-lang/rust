#![feature(core_intrinsics, portable_simd)]

use std::intrinsics::simd::*;
use std::simd::*;

fn main() {
    unsafe {
        let mut buf = [0u32; 5];
        //~v ERROR: accessing memory with alignment
        simd_masked_store::<_, _, _, { SimdAlign::Element }>(
            i32x4::splat(-1),
            // This is not i32-aligned
            buf.as_mut_ptr().byte_offset(1),
            i32x4::splat(0),
        );
    }
}

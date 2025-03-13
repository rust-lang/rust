//@ build-fail

#![feature(core_intrinsics)]

use std::intrinsics::simd::simd_add;

fn main() {
    unsafe {
        simd_add(0, 1) //~ ERROR E0511
    };
}

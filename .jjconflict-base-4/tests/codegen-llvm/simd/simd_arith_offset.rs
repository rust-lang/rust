//@ compile-flags: -C no-prepopulate-passes
//@ only-64bit (because the LLVM type of i64 for usize shows up)
//

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use std::intrinsics::simd::simd_arith_offset;

use minisimd::*;

/// A vector of *const T.
pub type SimdConstPtr<T, const LANES: usize> = Simd<*const T, LANES>;

// CHECK-LABEL: smoke
#[no_mangle]
pub fn smoke(ptrs: SimdConstPtr<u8, 8>, offsets: Simd<usize, 8>) -> SimdConstPtr<u8, 8> {
    // CHECK: getelementptr i8, <8 x ptr> %0, <8 x i64> %1
    unsafe { simd_arith_offset(ptrs, offsets) }
}

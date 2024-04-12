//@ compile-flags: -C no-prepopulate-passes
//@ only-64bit (because the LLVM type of i64 for usize shows up)
//

#![crate_type = "lib"]
#![feature(repr_simd, intrinsics)]

extern "rust-intrinsic" {
    pub(crate) fn simd_arith_offset<T, U>(ptrs: T, offsets: U) -> T;
}

/// A vector of *const T.
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub struct SimdConstPtr<T, const LANES: usize>([*const T; LANES]);

#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub struct Simd<T, const LANES: usize>([T; LANES]);

// CHECK-LABEL: u8_offset
#[no_mangle]
pub fn u8_offset(ptrs: SimdConstPtr<u8, 8>, offsets: Simd<usize, 8>) -> SimdConstPtr<u8, 8> {
    // CHECK: getelementptr [1 x i8], <8 x ptr> %0, <8 x i64> %1
    unsafe { simd_arith_offset(ptrs, offsets) }
}

// CHECK-LABEL: u64_offset
#[no_mangle]
pub fn u64_offset(ptrs: SimdConstPtr<u64, 8>, offsets: Simd<usize, 8>) -> SimdConstPtr<u64, 8> {
    // CHECK: getelementptr [8 x i8], <8 x ptr> %0, <8 x i64> %1
    unsafe { simd_arith_offset(ptrs, offsets) }
}

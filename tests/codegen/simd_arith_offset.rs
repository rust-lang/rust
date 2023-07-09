// compile-flags: -C no-prepopulate-passes
// only-64bit (because the LLVM type of i64 for usize shows up)
//

#![crate_type = "lib"]
#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    pub(crate) fn simd_arith_offset<T, U>(ptrs: T, offsets: U) -> T;
}

/// A vector of *const T.
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub struct SimdConstPtr<T, const LANES: usize>([*const T; LANES]);

#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub struct Simd<T, const LANES: usize>([T; LANES]);

// CHECK-LABEL: smoke
#[no_mangle]
pub fn smoke(ptrs: SimdConstPtr<u8, 8>, offsets: Simd<usize, 8>) -> SimdConstPtr<u8, 8> {
    // CHECK: getelementptr i8, <8 x {{i8\*|ptr}}> %0, <8 x i64> %1
    unsafe { simd_arith_offset(ptrs, offsets) }
}

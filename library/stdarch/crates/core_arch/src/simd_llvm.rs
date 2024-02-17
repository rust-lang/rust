//! LLVM's SIMD platform intrinsics

// This re-export is temporary; once libcore has all the required intrinsics, we can
// remove this entire file.
#[allow(unused_imports)]
pub use crate::intrinsics::simd::*;

extern "platform-intrinsic" {
    #[rustc_const_unstable(feature = "const_simd_insert", issue = "none")]
    pub fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
    #[rustc_const_unstable(feature = "const_simd_extract", issue = "none")]
    pub fn simd_extract<T, U>(x: T, idx: u32) -> U;

    pub fn simd_reduce_add_unordered<T, U>(x: T) -> U;
    pub fn simd_reduce_mul_unordered<T, U>(x: T) -> U;
}

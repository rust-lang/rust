//! LLVM's SIMD platform intrinsics

extern "platform-intrinsic" {
    //pub fn simd_select_bitmask
    pub fn simd_eq<T, U>(x: T, y: T) -> U;
    pub fn simd_ne<T, U>(x: T, y: T) -> U;
    pub fn simd_lt<T, U>(x: T, y: T) -> U;
    pub fn simd_le<T, U>(x: T, y: T) -> U;
    pub fn simd_gt<T, U>(x: T, y: T) -> U;
    pub fn simd_ge<T, U>(x: T, y: T) -> U;

    pub fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
    pub fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
    pub fn simd_shuffle8<T, U>(x: T, y: T, idx: [u32; 8]) -> U;
    pub fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
    pub fn simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U;
    pub fn simd_shuffle64<T, U>(x: T, y: T, idx: [u32; 64]) -> U;
    pub fn simd_shuffle128<T, U>(x: T, y: T, idx: [u32; 128]) -> U;

    #[rustc_const_unstable(feature = "const_simd_insert", issue = "none")]
    pub fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
    #[rustc_const_unstable(feature = "const_simd_extract", issue = "none")]
    pub fn simd_extract<T, U>(x: T, idx: u32) -> U;
    //pub fn simd_select
    pub fn simd_bitmask<T, U>(x: T) -> U;

    pub fn simd_cast<T, U>(x: T) -> U;

    pub fn simd_add<T>(x: T, y: T) -> T;
    pub fn simd_sub<T>(x: T, y: T) -> T;
    pub fn simd_mul<T>(x: T, y: T) -> T;
    pub fn simd_div<T>(x: T, y: T) -> T;
    pub fn simd_shl<T>(x: T, y: T) -> T;
    pub fn simd_shr<T>(x: T, y: T) -> T;
    pub fn simd_and<T>(x: T, y: T) -> T;
    pub fn simd_or<T>(x: T, y: T) -> T;
    pub fn simd_xor<T>(x: T, y: T) -> T;

    pub fn simd_neg<T>(x: T) -> T;

    pub fn simd_saturating_add<T>(x: T, y: T) -> T;
    pub fn simd_saturating_sub<T>(x: T, y: T) -> T;

    pub fn simd_gather<T, U, V>(values: T, pointers: U, mask: V) -> T;
    pub fn simd_scatter<T, U, V>(values: T, pointers: U, mask: V);

    pub fn simd_reduce_add_unordered<T, U>(x: T) -> U;
    pub fn simd_reduce_mul_unordered<T, U>(x: T) -> U;
    pub fn simd_reduce_add_ordered<T, U>(x: T, acc: U) -> U;
    pub fn simd_reduce_mul_ordered<T, U>(x: T, acc: U) -> U;
    pub fn simd_reduce_min<T, U>(x: T) -> U;
    pub fn simd_reduce_max<T, U>(x: T) -> U;
    pub fn simd_reduce_min_nanless<T, U>(x: T) -> U;
    pub fn simd_reduce_max_nanless<T, U>(x: T) -> U;
    pub fn simd_reduce_and<T, U>(x: T) -> U;
    pub fn simd_reduce_or<T, U>(x: T) -> U;
    pub fn simd_reduce_xor<T, U>(x: T) -> U;
    pub fn simd_reduce_all<T>(x: T) -> bool;
    pub fn simd_reduce_any<T>(x: T) -> bool;

    pub fn simd_select<M, T>(m: M, a: T, b: T) -> T;
    pub fn simd_select_bitmask<M, T>(m: M, a: T, b: T) -> T;

    pub fn simd_fmin<T>(a: T, b: T) -> T;
    pub fn simd_fmax<T>(a: T, b: T) -> T;

    pub fn simd_fsqrt<T>(a: T) -> T;
    pub fn simd_fsin<T>(a: T) -> T;
    pub fn simd_fcos<T>(a: T) -> T;
    pub fn simd_fabs<T>(a: T) -> T;
    pub fn simd_floor<T>(a: T) -> T;
    pub fn simd_ceil<T>(a: T) -> T;
    pub fn simd_fexp<T>(a: T) -> T;
    pub fn simd_fexp2<T>(a: T) -> T;
    pub fn simd_flog10<T>(a: T) -> T;
    pub fn simd_flog2<T>(a: T) -> T;
    pub fn simd_flog<T>(a: T) -> T;
    //pub fn simd_fpowi
    //pub fn simd_fpow
    pub fn simd_fma<T>(a: T, b: T, c: T) -> T;
}

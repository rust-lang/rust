//! Utilities used in testing the x86 intrinsics

use crate::core_arch::assert_eq_const as assert_eq;
use crate::core_arch::simd::*;
use crate::core_arch::x86::*;
use std::mem::transmute;

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(a.as_u32x4(), b.as_u32x4());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m128(a: __m128, b: __m128) {
    assert_eq!(a.as_f32x4(), b.as_f32x4());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m128d(a: __m128d, b: __m128d) {
    assert_eq!(a.as_f64x2(), b.as_f64x2());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m128h(a: __m128h, b: __m128h) {
    assert_eq!(a.as_f16x8(), b.as_f16x8());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(a.as_u32x8(), b.as_u32x8());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m256(a: __m256, b: __m256) {
    assert_eq!(a.as_f32x8(), b.as_f32x8());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m256d(a: __m256d, b: __m256d) {
    assert_eq!(a.as_f64x4(), b.as_f64x4());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m256h(a: __m256h, b: __m256h) {
    assert_eq!(a.as_f16x16(), b.as_f16x16());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m512i(a: __m512i, b: __m512i) {
    assert_eq!(a.as_i64x8(), b.as_i64x8());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m512(a: __m512, b: __m512) {
    assert_eq!(a.as_f32x16(), b.as_f32x16());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m512d(a: __m512d, b: __m512d) {
    assert_eq!(a.as_f64x8(), b.as_f64x8());
}

#[track_caller]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const fn assert_eq_m512h(a: __m512h, b: __m512h) {
    assert_eq!(a.as_f16x32(), b.as_f16x32());
}

#[target_feature(enable = "sse2")]
pub(crate) const fn get_m128d(a: __m128d, idx: usize) -> f64 {
    a.as_f64x2().extract(idx)
}

#[target_feature(enable = "sse")]
pub(crate) const fn get_m128(a: __m128, idx: usize) -> f32 {
    a.as_f32x4().extract(idx)
}

#[target_feature(enable = "avx")]
pub(crate) const fn get_m256d(a: __m256d, idx: usize) -> f64 {
    a.as_f64x4().extract(idx)
}

#[target_feature(enable = "avx")]
pub(crate) const fn get_m256(a: __m256, idx: usize) -> f32 {
    a.as_f32x8().extract(idx)
}

#[target_feature(enable = "avx512f")]
pub(crate) const fn get_m512(a: __m512, idx: usize) -> f32 {
    a.as_f32x16().extract(idx)
}

#[target_feature(enable = "avx512f")]
pub(crate) const fn get_m512d(a: __m512d, idx: usize) -> f64 {
    a.as_f64x8().extract(idx)
}

#[target_feature(enable = "avx512f")]
pub(crate) const fn get_m512i(a: __m512i, idx: usize) -> i64 {
    a.as_i64x8().extract(idx)
}

// not actually an intrinsic but useful in various tests as we ported from
// `i64x2::new` which is backwards from `_mm_set_epi64x`
#[target_feature(enable = "sse2")]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub const fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

// These intrinsics doesn't exist on x86 b/c it requires a 64-bit register,
// which doesn't exist on x86!
#[cfg(target_arch = "x86")]
mod x86_polyfill {
    use crate::core_arch::x86::*;
    use crate::intrinsics::simd::*;

    #[rustc_legacy_const_generics(2)]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub const fn _mm_insert_epi64<const INDEX: i32>(a: __m128i, val: i64) -> __m128i {
        static_assert_uimm_bits!(INDEX, 1);
        unsafe { transmute(simd_insert!(a.as_i64x2(), INDEX as u32, val)) }
    }

    #[target_feature(enable = "avx2")]
    #[rustc_legacy_const_generics(2)]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub const fn _mm256_insert_epi64<const INDEX: i32>(a: __m256i, val: i64) -> __m256i {
        static_assert_uimm_bits!(INDEX, 2);
        unsafe { transmute(simd_insert!(a.as_i64x4(), INDEX as u32, val)) }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_polyfill {
    pub use crate::core_arch::x86_64::{_mm_insert_epi64, _mm256_insert_epi64};
}
pub use self::x86_polyfill::*;

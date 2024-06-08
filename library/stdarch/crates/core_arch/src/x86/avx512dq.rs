use crate::{
    core_arch::{simd::*, x86::*},
    intrinsics::simd::*,
    mem::{self, transmute},
    ptr,
};

// And //

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm_mask_and_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let and = _mm_and_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, and, src.as_f64x2()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm_maskz_and_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let and = _mm_and_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, and, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm256_mask_and_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let and = _mm256_and_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, and, src.as_f64x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm256_maskz_and_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let and = _mm256_and_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, and, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm512_and_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_and(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b)))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm512_mask_and_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let and = _mm512_and_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, and, src.as_f64x8()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
// FIXME: should be `vandpd` instruction.
pub unsafe fn _mm512_maskz_and_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let and = _mm512_and_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, and, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm_mask_and_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let and = _mm_and_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, and, src.as_f32x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm_maskz_and_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let and = _mm_and_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, and, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm256_mask_and_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let and = _mm256_and_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, and, src.as_f32x8()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm256_maskz_and_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let and = _mm256_and_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, and, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm512_and_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_and(
        transmute::<_, u32x16>(a),
        transmute::<_, u32x16>(b),
    ))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm512_mask_and_ps(src: __m512, k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let and = _mm512_and_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, and, src.as_f32x16()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm512_maskz_and_ps(k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let and = _mm512_and_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, and, zero))
}

// Andnot

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm_mask_andnot_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let andnot = _mm_andnot_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x2()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm_maskz_andnot_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let andnot = _mm_andnot_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, andnot, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm256_mask_andnot_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let andnot = _mm256_andnot_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm256_maskz_andnot_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let andnot = _mm256_andnot_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, andnot, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm512_andnot_pd(a: __m512d, b: __m512d) -> __m512d {
    _mm512_and_pd(_mm512_xor_pd(a, transmute(_mm512_set1_epi64(-1))), b)
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm512_mask_andnot_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let andnot = _mm512_andnot_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x8()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
// FIXME: should be `vandnpd` instruction.
pub unsafe fn _mm512_maskz_andnot_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let andnot = _mm512_andnot_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, andnot, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm_mask_andnot_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let andnot = _mm_andnot_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, andnot, src.as_f32x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm_maskz_andnot_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let andnot = _mm_andnot_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, andnot, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm256_mask_andnot_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let andnot = _mm256_andnot_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, andnot, src.as_f32x8()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm256_maskz_andnot_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let andnot = _mm256_andnot_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, andnot, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm512_andnot_ps(a: __m512, b: __m512) -> __m512 {
    _mm512_and_ps(_mm512_xor_ps(a, transmute(_mm512_set1_epi32(-1))), b)
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm512_mask_andnot_ps(src: __m512, k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let andnot = _mm512_andnot_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, andnot, src.as_f32x16()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm512_maskz_andnot_ps(k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let andnot = _mm512_andnot_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, andnot, zero))
}

// Or

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm_mask_or_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let or = _mm_or_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, or, src.as_f64x2()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm_maskz_or_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let or = _mm_or_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, or, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm256_mask_or_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let or = _mm256_or_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, or, src.as_f64x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm256_maskz_or_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let or = _mm256_or_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, or, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm512_or_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_or(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b)))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm512_mask_or_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let or = _mm512_or_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, or, src.as_f64x8()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
// FIXME: should be `vorpd` instruction.
pub unsafe fn _mm512_maskz_or_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let or = _mm512_or_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, or, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm_mask_or_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let or = _mm_or_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, or, src.as_f32x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm_maskz_or_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let or = _mm_or_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, or, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm256_mask_or_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let or = _mm256_or_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, or, src.as_f32x8()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm256_maskz_or_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let or = _mm256_or_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, or, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm512_or_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_or(
        transmute::<_, u32x16>(a),
        transmute::<_, u32x16>(b),
    ))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm512_mask_or_ps(src: __m512, k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let or = _mm512_or_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, or, src.as_f32x16()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm512_maskz_or_ps(k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let or = _mm512_or_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, or, zero))
}

// Xor

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm_mask_xor_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let xor = _mm_xor_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, xor, src.as_f64x2()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm_maskz_xor_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let xor = _mm_xor_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, xor, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm256_mask_xor_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let xor = _mm256_xor_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, xor, src.as_f64x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm256_maskz_xor_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let xor = _mm256_xor_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, xor, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm512_xor_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_xor(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b)))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm512_mask_xor_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let xor = _mm512_xor_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, xor, src.as_f64x8()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
// FIXME: should be `vxorpd` instruction.
pub unsafe fn _mm512_maskz_xor_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let xor = _mm512_xor_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, xor, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm_mask_xor_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let xor = _mm_xor_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, xor, src.as_f32x4()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm_maskz_xor_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let xor = _mm_xor_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, xor, zero))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm256_mask_xor_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let xor = _mm256_xor_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, xor, src.as_f32x8()))
}

#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm256_maskz_xor_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let xor = _mm256_xor_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, xor, zero))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_xor_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_xor(
        transmute::<_, u32x16>(a),
        transmute::<_, u32x16>(b),
    ))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_mask_xor_ps(src: __m512, k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let xor = _mm512_xor_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, xor, src.as_f32x16()))
}

#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_maskz_xor_ps(k: __mmask8, a: __m512, b: __m512) -> __m512 {
    let xor = _mm512_xor_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, xor, zero))
}

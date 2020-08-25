use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Computes the absolute values of packed 32-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33&text=_mm512_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_abs_epi32(a: __m512i) -> __m512i {
    let a = a.as_i32x16();
    // all-0 is a properly initialized i32x16
    let zero: i32x16 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i32x16 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Computes the absolute value of packed 32-bit integers in `a`, and store the
/// unsigned results in `dst` using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33&text=_mm512_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_mask_abs_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, abs, src.as_i32x16()))
}

/// Computes the absolute value of packed 32-bit integers in `a`, and store the
/// unsigned results in `dst` using zeromask `k` (elements are zeroed out when
/// the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33,34,35,35&text=_mm512_maskz_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_maskz_abs_epi32(k: __mmask16, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 64-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_abs_epi64&expand=48)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsq))]
pub unsafe fn _mm512_abs_epi64(a: __m512i) -> __m512i {
    let a = a.as_i64x8();
    // all-0 is a properly initialized i64x8
    let zero: i64x8 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i64x8 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Compute the absolute value of packed signed 64-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_abs_epi64&expand=49)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsq))]
pub unsafe fn _mm512_mask_abs_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, abs, src.as_i64x8()))
}

/// Compute the absolute value of packed signed 64-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_abs_epi64&expand=50)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsq))]
pub unsafe fn _mm512_maskz_abs_epi64(k: __mmask8, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Returns vector of type `__m512d` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_pd() -> __m512d {
    // All-0 is a properly initialized __m512d
    mem::zeroed()
}

/// Returns vector of type `__m512d` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_ps() -> __m512 {
    // All-0 is a properly initialized __m512
    mem::zeroed()
}

/// Returns vector of type `__m512i` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_si512() -> __m512i {
    // All-0 is a properly initialized __m512i
    mem::zeroed()
}

/// Sets packed 32-bit integers in `dst` with the supplied values in reverse
/// order.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_setr_epi32(
    e15: i32,
    e14: i32,
    e13: i32,
    e12: i32,
    e11: i32,
    e10: i32,
    e9: i32,
    e8: i32,
    e7: i32,
    e6: i32,
    e5: i32,
    e4: i32,
    e3: i32,
    e2: i32,
    e1: i32,
    e0: i32,
) -> __m512i {
    let r = i32x16(
        e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
    );
    transmute(r)
}

/// Gather double-precision (64-bit) floating-point elements from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32gather_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherdpd, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i32gather_pd(offsets: __m256i, slice: *const u8, scale: i32) -> __m512d {
    let zero = _mm512_setzero_pd().as_f64x8();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherdpd(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather double-precision (64-bit) floating-point elements from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32gather_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherdpd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32gather_pd(
    src: __m512d,
    mask: __mmask8,
    offsets: __m256i,
    slice: *const u8,
    scale: i32,
) -> __m512d {
    let src = src.as_f64x8();
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherdpd(src, slice, offsets, mask as i8, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather double-precision (64-bit) floating-point elements from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64gather_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherqpd, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i64gather_pd(offsets: __m512i, slice: *const u8, scale: i32) -> __m512d {
    let zero = _mm512_setzero_pd().as_f64x8();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherqpd(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather double-precision (64-bit) floating-point elements from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64gather_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherqpd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64gather_pd(
    src: __m512d,
    mask: __mmask8,
    offsets: __m512i,
    slice: *const u8,
    scale: i32,
) -> __m512d {
    let src = src.as_f64x8();
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherqpd(src, slice, offsets, mask as i8, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather single-precision (32-bit) floating-point elements from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64gather_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherqps, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i64gather_ps(offsets: __m512i, slice: *const u8, scale: i32) -> __m256 {
    let zero = _mm256_setzero_ps().as_f32x8();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherqps(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather single-precision (32-bit) floating-point elements from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64gather_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherqps, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64gather_ps(
    src: __m256,
    mask: __mmask8,
    offsets: __m512i,
    slice: *const u8,
    scale: i32,
) -> __m256 {
    let src = src.as_f32x8();
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherqps(src, slice, offsets, mask as i8, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather single-precision (32-bit) floating-point elements from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32gather_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherdps, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i32gather_ps(offsets: __m512i, slice: *const u8, scale: i32) -> __m512 {
    let zero = _mm512_setzero_ps().as_f32x16();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherdps(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather single-precision (32-bit) floating-point elements from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32gather_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgatherdps, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32gather_ps(
    src: __m512,
    mask: __mmask16,
    offsets: __m512i,
    slice: *const u8,
    scale: i32,
) -> __m512 {
    let src = src.as_f32x16();
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vgatherdps(src, slice, offsets, mask as i16, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 32-bit integers from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32gather_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherdd, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i32gather_epi32(offsets: __m512i, slice: *const u8, scale: i32) -> __m512i {
    let zero = _mm512_setzero_si512().as_i32x16();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherdd(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 32-bit integers from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32gather_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherdd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32gather_epi32(
    src: __m512i,
    mask: __mmask16,
    offsets: __m512i,
    slice: *const u8,
    scale: i32,
) -> __m512i {
    let src = src.as_i32x16();
    let mask = mask as i16;
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherdd(src, slice, offsets, mask, $imm8)
        };
    }
    let r = constify_imm8!(scale, call);
    transmute(r)
}

/// Gather 64-bit integers from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32gather_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherdq, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i32gather_epi64(offsets: __m256i, slice: *const u8, scale: i32) -> __m512i {
    let zero = _mm512_setzero_si512().as_i64x8();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherdq(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 64-bit integers from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32gather_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherdq, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32gather_epi64(
    src: __m512i,
    mask: __mmask8,
    offsets: __m256i,
    slice: *const u8,
    scale: i32,
) -> __m512i {
    let src = src.as_i64x8();
    let mask = mask as i8;
    let slice = slice as *const i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherdq(src, slice, offsets, mask, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 64-bit integers from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64gather_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherqq, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i64gather_epi64(offsets: __m512i, slice: *const u8, scale: i32) -> __m512i {
    let zero = _mm512_setzero_si512().as_i64x8();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherqq(zero, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 64-bit integers from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64gather_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherqq, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64gather_epi64(
    src: __m512i,
    mask: __mmask8,
    offsets: __m512i,
    slice: *const u8,
    scale: i32,
) -> __m512i {
    let src = src.as_i64x8();
    let mask = mask as i8;
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherqq(src, slice, offsets, mask, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 32-bit integers from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64gather_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherqd, scale = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_i64gather_epi32(offsets: __m512i, slice: *const u8, scale: i32) -> __m256i {
    let zeros = _mm256_setzero_si256().as_i32x8();
    let neg_one = -1;
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherqd(zeros, slice, offsets, neg_one, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Gather 32-bit integers from memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64gather_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpgatherqd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64gather_epi32(
    src: __m256i,
    mask: __mmask8,
    offsets: __m512i,
    slice: *const u8,
    scale: i32,
) -> __m256i {
    let src = src.as_i32x8();
    let mask = mask as i8;
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpgatherqd(src, slice, offsets, mask, $imm8)
        };
    }
    let r = constify_imm8_gather!(scale, call);
    transmute(r)
}

/// Scatter double-precision (64-bit) floating-point elements from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32scatter_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterdpd, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i32scatter_pd(slice: *mut u8, offsets: __m256i, src: __m512d, scale: i32) {
    let src = src.as_f64x8();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterdpd(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter double-precision (64-bit) floating-point elements from src into memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32scatter_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterdpd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32scatter_pd(
    slice: *mut u8,
    mask: __mmask8,
    offsets: __m256i,
    src: __m512d,
    scale: i32,
) {
    let src = src.as_f64x8();
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterdpd(slice, mask as i8, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter double-precision (64-bit) floating-point elements from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64scatter_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterqpd, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i64scatter_pd(slice: *mut u8, offsets: __m512i, src: __m512d, scale: i32) {
    let src = src.as_f64x8();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterqpd(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter double-precision (64-bit) floating-point elements from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64scatter_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterqpd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64scatter_pd(
    slice: *mut u8,
    mask: __mmask8,
    offsets: __m512i,
    src: __m512d,
    scale: i32,
) {
    let src = src.as_f64x8();
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterqpd(slice, mask as i8, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter single-precision (32-bit) floating-point elements from memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32scatter_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterdps, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i32scatter_ps(slice: *mut u8, offsets: __m512i, src: __m512, scale: i32) {
    let src = src.as_f32x16();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterdps(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter single-precision (32-bit) floating-point elements from src into memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32scatter_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterdps, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32scatter_ps(
    slice: *mut u8,
    mask: __mmask16,
    offsets: __m512i,
    src: __m512,
    scale: i32,
) {
    let src = src.as_f32x16();
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterdps(slice, mask as i16, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter single-precision (32-bit) floating-point elements from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64scatter_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterqps, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i64scatter_ps(slice: *mut u8, offsets: __m512i, src: __m256, scale: i32) {
    let src = src.as_f32x8();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterqps(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter single-precision (32-bit) floating-point elements from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64scatter_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vscatterqps, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64scatter_ps(
    slice: *mut u8,
    mask: __mmask8,
    offsets: __m512i,
    src: __m256,
    scale: i32,
) {
    let src = src.as_f32x8();
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vscatterqps(slice, mask as i8, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 64-bit integers from src into memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i32scatter_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterdq, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i32scatter_epi64(slice: *mut u8, offsets: __m256i, src: __m512i, scale: i32) {
    let src = src.as_i64x8();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterdq(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 64-bit integers from src into memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32scatter_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterdq, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32scatter_epi64(
    slice: *mut u8,
    mask: __mmask8,
    offsets: __m256i,
    src: __m512i,
    scale: i32,
) {
    let src = src.as_i64x8();
    let mask = mask as i8;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterdq(slice, mask, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 64-bit integers from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64scatter_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterqq, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i64scatter_epi64(slice: *mut u8, offsets: __m512i, src: __m512i, scale: i32) {
    let src = src.as_i64x8();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterqq(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 64-bit integers from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64scatter_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterqq, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64scatter_epi64(
    slice: *mut u8,
    mask: __mmask8,
    offsets: __m512i,
    src: __m512i,
    scale: i32,
) {
    let src = src.as_i64x8();
    let mask = mask as i8;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterqq(slice, mask, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 32-bit integers from src into memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64scatter_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterdd, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i32scatter_epi32(slice: *mut u8, offsets: __m512i, src: __m512i, scale: i32) {
    let src = src.as_i32x16();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterdd(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 32-bit integers from src into memory using 32-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i32scatter_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterdd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i32scatter_epi32(
    slice: *mut u8,
    mask: __mmask16,
    offsets: __m512i,
    src: __m512i,
    scale: i32,
) {
    let src = src.as_i32x16();
    let mask = mask as i16;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i32x16();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterdd(slice, mask, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 32-bit integers from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_i64scatter_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterqd, scale = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_i64scatter_epi32(slice: *mut u8, offsets: __m512i, src: __m256i, scale: i32) {
    let src = src.as_i32x8();
    let neg_one = -1;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterqd(slice, neg_one, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Scatter 32-bit integers from src into memory using 64-bit indices.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_i64scatter_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpscatterqd, scale = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_i64scatter_epi32(
    slice: *mut u8,
    mask: __mmask8,
    offsets: __m512i,
    src: __m256i,
    scale: i32,
) {
    let src = src.as_i32x8();
    let mask = mask as i8;
    let slice = slice as *mut i8;
    let offsets = offsets.as_i64x8();
    macro_rules! call {
        ($imm8:expr) => {
            vpscatterqd(slice, mask, offsets, src, $imm8)
        };
    }
    constify_imm8_gather!(scale, call);
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rol_epi32&expand=4685)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_rol_epi32(a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    transmute(vprold(a.as_i32x16(), imm8))
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rol_epi32&expand=4683)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_rol_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let rol = vprold(a.as_i32x16(), imm8);
    transmute(simd_select_bitmask(k, rol, src.as_i32x16()))
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rol_epi32&expand=4684)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_rol_epi32(k: __mmask16, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let rol = vprold(a.as_i32x16(), imm8);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, rol, zero))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_ror_epi32&expand=4721)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 233))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_ror_epi32(a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    transmute(vprord(a.as_i32x16(), imm8))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_ror_epi32&expand=4719)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 123))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_ror_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let ror = vprord(a.as_i32x16(), imm8);
    transmute(simd_select_bitmask(k, ror, src.as_i32x16()))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_ror_epi32&expand=4720)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 123))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_ror_epi32(k: __mmask16, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let ror = vprord(a.as_i32x16(), imm8);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, ror, zero))
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rol_epi64&expand=4694)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_rol_epi64(a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    transmute(vprolq(a.as_i64x8(), imm8))
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rol_epi64&expand=4692)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_rol_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let rol = vprolq(a.as_i64x8(), imm8);
    transmute(simd_select_bitmask(k, rol, src.as_i64x8()))
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rol_epi64&expand=4693)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_rol_epi64(k: __mmask8, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let rol = vprolq(a.as_i64x8(), imm8);
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, rol, zero))
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_ror_epi64&expand=4730)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 15))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_ror_epi64(a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    transmute(vprorq(a.as_i64x8(), imm8))
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_ror_epi64&expand=4728)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 15))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_ror_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let ror = vprorq(a.as_i64x8(), imm8);
    transmute(simd_select_bitmask(k, ror, src.as_i64x8()))
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_ror_epi64&expand=4729)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 15))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_ror_epi64(k: __mmask8, a: __m512i, imm8: i32) -> __m512i {
    assert!(imm8 >= 0 && imm8 <= 255);
    let ror = vprorq(a.as_i64x8(), imm8);
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, ror, zero))
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_slli_epi32&expand=5310)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_slli_epi32(a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    transmute(vpsllid(a.as_i32x16(), imm8))
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_slli_epi32&expand=5308)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_slli_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsllid(a.as_i32x16(), imm8);
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_slli_epi32&expand=5309)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_slli_epi32(k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsllid(a.as_i32x16(), imm8);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srli_epi32&expand=5522)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srli_epi32(a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    transmute(vpsrlid(a.as_i32x16(), imm8))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srli_epi32&expand=5520)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srli_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsrlid(a.as_i32x16(), imm8);
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srli_epi32&expand=5521)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srli_epi32(k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsrlid(a.as_i32x16(), imm8);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_slli_epi64&expand=5319)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq, imm8 = 5))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_slli_epi64(a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    transmute(vpslliq(a.as_i64x8(), imm8))
}

/// Shift packed 64-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_slli_epi64&expand=5317)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_slli_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpslliq(a.as_i64x8(), imm8);
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_slli_epi64&expand=5318)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_slli_epi64(k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpslliq(a.as_i64x8(), imm8);
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srli_epi64&expand=5531)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srli_epi64(a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    transmute(vpsrliq(a.as_i64x8(), imm8))
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srli_epi64&expand=5529)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srli_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsrliq(a.as_i64x8(), imm8);
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srli_epi64&expand=5530)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srli_epi64(k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsrliq(a.as_i64x8(), imm8);
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a left by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sll_epi32&expand=5280)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm512_sll_epi32(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpslld(a.as_i32x16(), count.as_i32x4()))
}

/// Shift packed 32-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sll_epi32&expand=5278)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm512_mask_sll_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sll_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sll_epi32&expand=5279)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm512_maskz_sll_epi32(k: __mmask16, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sll_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srl_epi32&expand=5492)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm512_srl_epi32(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsrld(a.as_i32x16(), count.as_i32x4()))
}

/// Shift packed 32-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srl_epi32&expand=5490)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm512_mask_srl_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_srl_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srl_epi32&expand=5491)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm512_maskz_srl_epi32(k: __mmask16, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_srl_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a left by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sll_epi64&expand=5289)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq))]
pub unsafe fn _mm512_sll_epi64(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsllq(a.as_i64x8(), count.as_i64x2()))
}

/// Shift packed 64-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sll_epi64&expand=5287)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq))]
pub unsafe fn _mm512_mask_sll_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sll_epi64(a, count).as_i64x8();
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sll_epi64&expand=5288)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq))]
pub unsafe fn _mm512_maskz_sll_epi64(k: __mmask8, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sll_epi64(a, count).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a right by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srl_epi64&expand=5501)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq))]
pub unsafe fn _mm512_srl_epi64(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsrlq(a.as_i64x8(), count.as_i64x2()))
}

/// Shift packed 64-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srl_epi64&expand=5499)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq))]
pub unsafe fn _mm512_mask_srl_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_srl_epi64(a, count).as_i64x8();
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sll_epi64&expand=5288)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq))]
pub unsafe fn _mm512_maskz_srl_epi64(k: __mmask8, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_srl_epi64(a, count).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sra_epi32&expand=5407)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm512_sra_epi32(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsrad(a.as_i32x16(), count.as_i32x4()))
}

/// Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sra_epi32&expand=5405)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm512_mask_sra_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sra_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sra_epi32&expand=5406)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm512_maskz_sra_epi32(k: __mmask16, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sra_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a right by count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sra_epi64&expand=5416)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq))]
pub unsafe fn _mm512_sra_epi64(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsraq(a.as_i64x8(), count.as_i64x2()))
}

/// Shift packed 64-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sra_epi64&expand=5414)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq))]
pub unsafe fn _mm512_mask_sra_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sra_epi64(a, count).as_i64x8();
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sra_epi64&expand=5415)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq))]
pub unsafe fn _mm512_maskz_sra_epi64(k: __mmask8, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sra_epi64(a, count).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srai_epi32&expand=5436)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srai_epi32(a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    transmute(vpsraid(a.as_i32x16(), imm8))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srai_epi32&expand=5434)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srai_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsraid(a.as_i32x16(), imm8);
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srai_epi32&expand=5435)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srai_epi32(k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsraid(a.as_i32x16(), imm8);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srai_epi64&expand=5445)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srai_epi64(a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    transmute(vpsraiq(a.as_i64x8(), imm8))
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srai_epi64&expand=5443)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srai_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsraiq(a.as_i64x8(), imm8);
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srai_epi64&expand=5444)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srai_epi64(k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    assert!(imm8 <= 255);
    let shf = vpsraiq(a.as_i64x8(), imm8);
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srav_epi32&expand=5465)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm512_srav_epi32(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsravd(a.as_i32x16(), count.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srav_epi32&expand=5463)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm512_mask_srav_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srav_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srav_epi32&expand=5464)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm512_maskz_srav_epi32(k: __mmask16, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srav_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srav_epi64&expand=5474)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravq))]
pub unsafe fn _mm512_srav_epi64(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsravq(a.as_i64x8(), count.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srav_epi64&expand=5472)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravq))]
pub unsafe fn _mm512_mask_srav_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srav_epi64(a, count).as_i64x8();
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srav_epi64&expand=5473)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravq))]
pub unsafe fn _mm512_maskz_srav_epi64(k: __mmask8, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srav_epi64(a, count).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in the corresponding element of b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rolv_epi32&expand=4703)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolvd))]
pub unsafe fn _mm512_rolv_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vprolvd(a.as_i32x16(), b.as_i32x16()))
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in the corresponding element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rolv_epi32&expand=4701)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolvd))]
pub unsafe fn _mm512_mask_rolv_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let rol = _mm512_rolv_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, rol, src.as_i32x16()))
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in the corresponding element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rolv_epi32&expand=4702)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolvd))]
pub unsafe fn _mm512_maskz_rolv_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let rol = _mm512_rolv_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, rol, zero))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in the corresponding element of b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rorv_epi32&expand=4739)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorvd))]
pub unsafe fn _mm512_rorv_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vprorvd(a.as_i32x16(), b.as_i32x16()))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in the corresponding element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rorv_epi32&expand=4737)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorvd))]
pub unsafe fn _mm512_mask_rorv_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let ror = _mm512_rorv_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, ror, src.as_i32x16()))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in the corresponding element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rorv_epi32&expand=4738)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorvd))]
pub unsafe fn _mm512_maskz_rorv_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let ror = _mm512_rorv_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, ror, zero))
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in the corresponding element of b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rolv_epi64&expand=4712)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolvq))]
pub unsafe fn _mm512_rolv_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vprolvq(a.as_i64x8(), b.as_i64x8()))
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in the corresponding element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rolv_epi64&expand=4710)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolvq))]
pub unsafe fn _mm512_mask_rolv_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let rol = _mm512_rolv_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, rol, src.as_i64x8()))
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in the corresponding element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rolv_epi64&expand=4711)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolvq))]
pub unsafe fn _mm512_maskz_rolv_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let rol = _mm512_rolv_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, rol, zero))
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in the corresponding element of b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rorv_epi64&expand=4748)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorvq))]
pub unsafe fn _mm512_rorv_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vprorvq(a.as_i64x8(), b.as_i64x8()))
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in the corresponding element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rorv_epi64&expand=4746)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorvq))]
pub unsafe fn _mm512_mask_rorv_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let ror = _mm512_rorv_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, ror, src.as_i64x8()))
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in the corresponding element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rorv_epi64&expand=4747)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorvq))]
pub unsafe fn _mm512_maskz_rorv_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let ror = _mm512_rorv_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, ror, zero))
}

/// Shift packed 32-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sllv_epi32&expand=5342)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm512_sllv_epi32(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsllvd(a.as_i32x16(), count.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sllv_epi32&expand=5340)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm512_mask_sllv_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_sllv_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sllv_epi32&expand=5341)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm512_maskz_sllv_epi32(k: __mmask16, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_sllv_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srlv_epi32&expand=5554)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm512_srlv_epi32(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsrlvd(a.as_i32x16(), count.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srlv_epi32&expand=5552)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm512_mask_srlv_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srlv_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srlv_epi32&expand=5553)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm512_maskz_srlv_epi32(k: __mmask16, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srlv_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sllv_epi64&expand=5351)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvq))]
pub unsafe fn _mm512_sllv_epi64(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsllvq(a.as_i64x8(), count.as_i64x8()))
}

/// Shift packed 64-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sllv_epi64&expand=5349)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvq))]
pub unsafe fn _mm512_mask_sllv_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_sllv_epi64(a, count).as_i64x8();
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sllv_epi64&expand=5350)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvq))]
pub unsafe fn _mm512_maskz_sllv_epi64(k: __mmask8, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_sllv_epi64(a, count).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srlv_epi64&expand=5563)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvq))]
pub unsafe fn _mm512_srlv_epi64(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsrlvq(a.as_i64x8(), count.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mask_srlv_epi64&expand=5561)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvq))]
pub unsafe fn _mm512_mask_srlv_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srlv_epi64(a, count).as_i64x8();
    transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
}

/// Shift packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srlv_epi64&expand=5562)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvq))]
pub unsafe fn _mm512_maskz_srlv_epi64(k: __mmask8, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srlv_epi64(a, count).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Compute the bitwise AND of packed 32-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_and_epi32&expand=272)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_and_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_and(a.as_i32x16(), b.as_i32x16()))
}

/// Performs element-by-element bitwise AND between packed 32-bit integer elements of v2 and v3, storing the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_and_epi32&expand=273)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandd))]
pub unsafe fn _mm512_mask_and_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let and = _mm512_and_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, and, src.as_i32x16()))
}

/// Compute the bitwise AND of packed 32-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_and_epi32&expand=274)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandd))]
pub unsafe fn _mm512_maskz_and_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let and = _mm512_and_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of 512 bits (composed of packed 64-bit integers) in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_and_epi64&expand=279)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_and_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_and(a.as_i64x8(), b.as_i64x8()))
}

/// Compute the bitwise AND of packed 64-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_and_epi64&expand=280)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_mask_and_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let and = _mm512_and_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, and, src.as_i64x8()))
}

/// Compute the bitwise AND of packed 32-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_and_Epi32&expand=274)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_maskz_and_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let and = _mm512_and_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of 512 bits (representing integer data) in a and b, and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_and_si512&expand=302)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_and_si512(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_and(a.as_i32x16(), b.as_i32x16()))
}

/// Compute the bitwise OR of packed 32-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_or_epi32&expand=4042)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vporq))]
pub unsafe fn _mm512_or_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_or(a.as_i32x16(), b.as_i32x16()))
}

/// Compute the bitwise OR of packed 32-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_or_epi32&expand=4040)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpord))]
pub unsafe fn _mm512_mask_or_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let or = _mm512_or_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, or, src.as_i32x16()))
}

/// Compute the bitwise OR of packed 32-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_or_epi32&expand=4041)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpord))]
pub unsafe fn _mm512_maskz_or_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let or = _mm512_or_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of packed 64-bit integers in a and b, and store the resut in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_or_epi64&expand=4051)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vporq))]
pub unsafe fn _mm512_or_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_or(a.as_i64x8(), b.as_i64x8()))
}

/// Compute the bitwise OR of packed 64-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_or_epi64&expand=4049)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vporq))]
pub unsafe fn _mm512_mask_or_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let or = _mm512_or_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, or, src.as_i64x8()))
}

/// Compute the bitwise OR of packed 64-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_or_epi64&expand=4050)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vporq))]
pub unsafe fn _mm512_maskz_or_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let or = _mm512_or_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of 512 bits (representing integer data) in a and b, and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_or_si512&expand=4072)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vporq))]
pub unsafe fn _mm512_or_si512(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_or(a.as_i32x16(), b.as_i32x16()))
}

/// Compute the bitwise XOR of packed 32-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_xor_epi32&expand=6142)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxorq))]
pub unsafe fn _mm512_xor_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_xor(a.as_i32x16(), b.as_i32x16()))
}

/// Compute the bitwise XOR of packed 32-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_xor_epi32&expand=6140)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxord))]
pub unsafe fn _mm512_mask_xor_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let xor = _mm512_xor_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, xor, src.as_i32x16()))
}

/// Compute the bitwise XOR of packed 32-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_xor_epi32&expand=6141)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxord))]
pub unsafe fn _mm512_maskz_xor_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let xor = _mm512_xor_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of packed 64-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_xor_epi64&expand=6151)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxorq))]
pub unsafe fn _mm512_xor_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_xor(a.as_i64x8(), b.as_i64x8()))
}

/// Compute the bitwise XOR of packed 64-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_xor_epi64&expand=6149)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxorq))]
pub unsafe fn _mm512_mask_xor_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let xor = _mm512_xor_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, xor, src.as_i64x8()))
}

/// Compute the bitwise XOR of packed 64-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_xor_epi64&expand=6150)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxorq))]
pub unsafe fn _mm512_maskz_xor_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let xor = _mm512_xor_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of 512 bits (representing integer data) in a and b, and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_xor_si512&expand=6172)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxorq))]
pub unsafe fn _mm512_xor_si512(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_xor(a.as_i32x16(), b.as_i32x16()))
}

/// Compute the bitwise AND of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kand_mask16&expand=3212)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(and))] // generate normal and code instead of kandw
pub unsafe fn _kand_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(kandw(a, b))
}

/// Compute the bitwise AND of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kand&expand=3210)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(and))] // generate normal and code instead of kandw
pub unsafe fn _mm512_kand(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(kandw(a, b))
}

/// Compute the bitwise OR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kor_mask16&expand=3239)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(or))] // generate normal or code instead of korw
pub unsafe fn _kor_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(korw(a, b))
}

/// Compute the bitwise OR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kor&expand=3237)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(or))] // generate normal or code instead of korw
pub unsafe fn _mm512_kor(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(korw(a, b))
}

/// Compute the bitwise XOR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kxor_mask16&expand=3291)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(xor))] // generate normal xor code instead of kxorw
pub unsafe fn _kxor_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(kxorw(a, b))
}

/// Compute the bitwise XOR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kxor&expand=3289)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(xor))] // generate normal xor code instead of kxorw
pub unsafe fn _mm512_kxor(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(kxorw(a, b))
}

/// Sets packed 32-bit integers in `dst` with the supplied values.
///
/// [Intel's documentation]( https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,4909&text=_mm512_set_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set_ps(
    e0: f32,
    e1: f32,
    e2: f32,
    e3: f32,
    e4: f32,
    e5: f32,
    e6: f32,
    e7: f32,
    e8: f32,
    e9: f32,
    e10: f32,
    e11: f32,
    e12: f32,
    e13: f32,
    e14: f32,
    e15: f32,
) -> __m512 {
    _mm512_setr_ps(
        e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
    )
}

/// Sets packed 32-bit integers in `dst` with the supplied values in
/// reverse order.
///
/// [Intel's documentation]( https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,4909&text=_mm512_set_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_setr_ps(
    e0: f32,
    e1: f32,
    e2: f32,
    e3: f32,
    e4: f32,
    e5: f32,
    e6: f32,
    e7: f32,
    e8: f32,
    e9: f32,
    e10: f32,
    e11: f32,
    e12: f32,
    e13: f32,
    e14: f32,
    e15: f32,
) -> __m512 {
    let r = f32x16::new(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    );
    transmute(r)
}

/// Broadcast 64-bit float `a` to all elements of `dst`.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set1_pd(a: f64) -> __m512d {
    transmute(f64x8::splat(a))
}

/// Broadcast 32-bit float `a` to all elements of `dst`.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set1_ps(a: f32) -> __m512 {
    transmute(f32x16::splat(a))
}

/// Sets packed 32-bit integers in `dst` with the supplied values.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set_epi32(
    e15: i32,
    e14: i32,
    e13: i32,
    e12: i32,
    e11: i32,
    e10: i32,
    e9: i32,
    e8: i32,
    e7: i32,
    e6: i32,
    e5: i32,
    e4: i32,
    e3: i32,
    e2: i32,
    e1: i32,
    e0: i32,
) -> __m512i {
    _mm512_setr_epi32(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Broadcast 32-bit integer `a` to all elements of `dst`.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set1_epi32(a: i32) -> __m512i {
    transmute(i32x16::splat(a))
}

/// Broadcast 64-bit integer `a` to all elements of `dst`.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set1_epi64(a: i64) -> __m512i {
    transmute(i64x8::splat(a))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmplt_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_LT_OS)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmplt_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpnlt_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpnlt_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_NLT_US)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpnlt_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpnlt_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmple_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_LE_OS)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmple_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpnle_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpnle_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_NLE_US)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpnle_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpnle_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpeq_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpeq_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpneq_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpneq_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_cmp_ps_mask(a: __m512, b: __m512, op: i32) -> __mmask16 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr) => {
            vcmpps(
                a.as_f32x16(),
                b.as_f32x16(),
                $imm5,
                neg_one,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_ps_mask(m: __mmask16, a: __m512, b: __m512, op: i32) -> __mmask16 {
    macro_rules! call {
        ($imm5:expr) => {
            vcmpps(
                a.as_f32x16(),
                b.as_f32x16(),
                $imm5,
                m as i16,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_round_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2, 3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm512_cmp_round_ps_mask(a: __m512, b: __m512, op: i32, sae: i32) -> __mmask16 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmpps(a.as_f32x16(), b.as_f32x16(), $imm5, neg_one, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_round_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3, 4)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm512_mask_cmp_round_ps_mask(
    m: __mmask16,
    a: __m512,
    b: __m512,
    op: i32,
    sae: i32,
) -> __mmask16 {
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmpps(a.as_f32x16(), b.as_f32x16(), $imm5, m as i16, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b to see if neither is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpord_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_cmpord_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b to see if neither is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpord_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_mask_cmpord_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b to see if either is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpunord_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_cmpunord_ps_mask(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b to see if either is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpunord_ps_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_mask_cmpunord_ps_mask(m: __mmask16, a: __m512, b: __m512) -> __mmask16 {
    _mm512_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmplt_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_LT_OS)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmplt_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpnlt_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpnlt_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_NLT_US)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpnlt_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpnlt_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmple_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_LE_OS)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmple_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpnle_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpnle_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_NLE_US)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpnle_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpnle_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpeq_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpeq_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_cmpneq_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp))]
pub unsafe fn _mm512_mask_cmpneq_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_cmp_pd_mask(a: __m512d, b: __m512d, op: i32) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr) => {
            vcmppd(
                a.as_f64x8(),
                b.as_f64x8(),
                $imm5,
                neg_one,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_pd_mask(m: __mmask8, a: __m512d, b: __m512d, op: i32) -> __mmask8 {
    macro_rules! call {
        ($imm5:expr) => {
            vcmppd(
                a.as_f64x8(),
                b.as_f64x8(),
                $imm5,
                m as i8,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_round_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2, 3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm512_cmp_round_pd_mask(a: __m512d, b: __m512d, op: i32, sae: i32) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmppd(a.as_f64x8(), b.as_f64x8(), $imm5, neg_one, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_round_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3, 4)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm512_mask_cmp_round_pd_mask(
    m: __mmask8,
    a: __m512d,
    b: __m512d,
    op: i32,
    sae: i32,
) -> __mmask8 {
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmppd(a.as_f64x8(), b.as_f64x8(), $imm5, m as i8, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b to see if neither is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpord_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_cmpord_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b to see if neither is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpord_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_mask_cmpord_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b to see if either is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpunord_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_cmpunord_pd_mask(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b to see if either is NaN, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpunord_pd_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcmp, op = 0))]
pub unsafe fn _mm512_mask_cmpunord_pd_mask(m: __mmask8, a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_ss_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_cmp_ss_mask(a: __m128, b: __m128, op: i32) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr) => {
            vcmpss(a, b, $imm5, neg_one, _MM_FROUND_CUR_DIRECTION)
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector using zeromask m (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_ss_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_mask_cmp_ss_mask(m: __mmask8, a: __m128, b: __m128, op: i32) -> __mmask8 {
    macro_rules! call {
        ($imm5:expr) => {
            vcmpss(a, b, $imm5, m as i8, _MM_FROUND_CUR_DIRECTION)
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_round_ss_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2, 3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_cmp_round_ss_mask(a: __m128, b: __m128, op: i32, sae: i32) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmpss(a, b, $imm5, neg_one, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector using zeromask m (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_round_ss_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3, 4)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_mask_cmp_round_ss_mask(
    m: __mmask8,
    a: __m128,
    b: __m128,
    op: i32,
    sae: i32,
) -> __mmask8 {
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmpss(a, b, $imm5, m as i8, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_sd_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_cmp_sd_mask(a: __m128d, b: __m128d, op: i32) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr) => {
            vcmpsd(a, b, $imm5, neg_one, _MM_FROUND_CUR_DIRECTION)
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector using zeromask m (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_sd_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_mask_cmp_sd_mask(m: __mmask8, a: __m128d, b: __m128d, op: i32) -> __mmask8 {
    macro_rules! call {
        ($imm5:expr) => {
            vcmpsd(a, b, $imm5, m as i8, _MM_FROUND_CUR_DIRECTION)
        };
    }
    let r = constify_imm5!(op, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_round_sd_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2, 3)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_cmp_round_sd_mask(a: __m128d, b: __m128d, op: i32, sae: i32) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmpsd(a, b, $imm5, neg_one, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare the lower single-precision (32-bit) floating-point element in a and b based on the comparison operand specified by imm8, and store the result in a mask vector using zeromask m (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_round_sd_mask&expand=5236,755,757)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3, 4)]
#[cfg_attr(test, assert_instr(vcmp, op = 0, sae = 4))]
pub unsafe fn _mm_mask_cmp_round_sd_mask(
    m: __mmask8,
    a: __m128d,
    b: __m128d,
    op: i32,
    sae: i32,
) -> __mmask8 {
    macro_rules! call {
        ($imm5:expr, $imm4:expr) => {
            vcmpsd(a, b, $imm5, m as i8, $imm4)
        };
    }
    let r = constify_imm5_sae!(op, sae, call);
    transmute(r)
}

/// Compare packed unsigned 32-bit integers in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_lt(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmplt_epu32_mask(a, b) & m
}

/// Compare packed unsigned 32-bit integers in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpgt_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_gt(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpgt_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpgt_epu32_mask(a, b) & m
}

/// Compare packed unsigned 32-bit integers in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_le(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmple_epu32_mask(a, b) & m
}

/// Compare packed unsigned 32-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpge_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_ge(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpge_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpge_epu32_mask(a, b) & m
}

/// Compare packed unsigned 32-bit integers in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_eq(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpeq_epu32_mask(a, b) & m
}

/// Compare packed unsigned 32-bit integers in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_epu32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_ne(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_epu32_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpneq_epu32_mask(a, b) & m
}

/// Compare packed unsigned 32-bit integers in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epu32_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_cmp_epu32_mask(a: __m512i, b: __m512i, op: _MM_CMPINT_ENUM) -> __mmask16 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpud(a.as_i32x16(), b.as_i32x16(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed unsigned 32-bit integers in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epu32_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_epu32_mask(
    m: __mmask16,
    a: __m512i,
    b: __m512i,
    op: _MM_CMPINT_ENUM,
) -> __mmask16 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpud(a.as_i32x16(), b.as_i32x16(), $imm3, m as i16)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed unsigned 32-bit integers in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_lt(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmplt_epi32_mask(a, b) & m
}

/// Compare packed signed 32-bit integers in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpgt_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_gt(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpgt_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpgt_epi32_mask(a, b) & m
}

/// Compare packed signed 32-bit integers in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_le(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmple_epi32_mask(a, b) & m
}

/// Compare packed signed 32-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpge_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_ge(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpge_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpge_epi32_mask(a, b) & m
}

/// Compare packed signed 32-bit integers in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_eq(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpeq_epi32_mask(a, b) & m
}

/// Compare packed signed 32-bit integers in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_ne(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi32_mask(m: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpneq_epi32_mask(a, b) & m
}

/// Compare packed signed 32-bit integers in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epi32_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_cmp_epi32_mask(a: __m512i, b: __m512i, op: _MM_CMPINT_ENUM) -> __mmask16 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpd(a.as_i32x16(), b.as_i32x16(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed signed 32-bit integers in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epi32_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_epi32_mask(
    m: __mmask16,
    a: __m512i,
    b: __m512i,
    op: _MM_CMPINT_ENUM,
) -> __mmask16 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpd(a.as_i32x16(), b.as_i32x16(), $imm3, m as i16)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed unsigned 64-bit integers in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_lt(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpgt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_gt(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpgt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_le(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmple_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpge_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ge(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpge_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpge_epu64_mask(b, a) & m
}

/// Compare packed unsigned 64-bit integers in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_eq(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpeq_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ne(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_epu64_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpneq_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epu64_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_cmp_epu64_mask(a: __m512i, b: __m512i, op: _MM_CMPINT_ENUM) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuq(a.as_i64x8(), b.as_i64x8(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed unsigned 64-bit integers in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epu64_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_epu64_mask(
    m: __mmask8,
    a: __m512i,
    b: __m512i,
    op: _MM_CMPINT_ENUM,
) -> __mmask8 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuq(a.as_i64x8(), b.as_i64x8(), $imm3, m as i8)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed signed 64-bit integers in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_lt(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpgt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_gt(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpgt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_le(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmple_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpge_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ge(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpge_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpge_epi64_mask(b, a) & m
}

/// Compare packed signed 64-bit integers in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_eq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpeq_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ne(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpneq_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epi64_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_cmp_epi64_mask(a: __m512i, b: __m512i, op: _MM_CMPINT_ENUM) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpq(a.as_i64x8(), b.as_i64x8(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed signed 64-bit integers in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epi64_mask)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_epi64_mask(
    m: __mmask8,
    a: __m512i,
    b: __m512i,
    op: _MM_CMPINT_ENUM,
) -> __mmask8 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpq(a.as_i64x8(), b.as_i64x8(), $imm3, m as i8)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Returns vector of type `__m512d` with undefined elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_undefined_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_undefined_pd() -> __m512d {
    _mm512_set1_pd(0.0)
}

/// Returns vector of type `__m512` with undefined elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_undefined_ps)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_undefined_ps() -> __m512 {
    _mm512_set1_ps(0.0)
}

/// Loads 512-bits (composed of 8 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))]
pub unsafe fn _mm512_loadu_pd(mem_addr: *const f64) -> __m512d {
    ptr::read_unaligned(mem_addr as *const __m512d)
}

/// Stores 512-bits (composed of 8 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))]
pub unsafe fn _mm512_storeu_pd(mem_addr: *mut f64, a: __m512d) {
    ptr::write_unaligned(mem_addr as *mut __m512d, a);
}

/// Loads 512-bits (composed of 16 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))]
pub unsafe fn _mm512_loadu_ps(mem_addr: *const f32) -> __m512 {
    ptr::read_unaligned(mem_addr as *const __m512)
}

/// Stores 512-bits (composed of 16 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm512_storeu_ps(mem_addr: *mut f32, a: __m512) {
    ptr::write_unaligned(mem_addr as *mut __m512, a);
}

/// Sets packed 64-bit integers in `dst` with the supplied values in
/// reverse order.
///
/// [Intel's documentation]( https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,4909&text=_mm512_set_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_setr_pd(
    e0: f64,
    e1: f64,
    e2: f64,
    e3: f64,
    e4: f64,
    e5: f64,
    e6: f64,
    e7: f64,
) -> __m512d {
    let r = f64x8::new(e0, e1, e2, e3, e4, e5, e6, e7);
    transmute(r)
}

/// Sets packed 64-bit integers in `dst` with the supplied values.
///
/// [Intel's documentation]( https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,4909&text=_mm512_set_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set_pd(
    e0: f64,
    e1: f64,
    e2: f64,
    e3: f64,
    e4: f64,
    e5: f64,
    e6: f64,
    e7: f64,
) -> __m512d {
    _mm512_setr_pd(e7, e6, e5, e4, e3, e2, e1, e0)
}

/// Equal
pub const _MM_CMPINT_EQ: _MM_CMPINT_ENUM = 0x00;
/// Less-than
pub const _MM_CMPINT_LT: _MM_CMPINT_ENUM = 0x01;
/// Less-than-or-equal
pub const _MM_CMPINT_LE: _MM_CMPINT_ENUM = 0x02;
/// False
pub const _MM_CMPINT_FALSE: _MM_CMPINT_ENUM = 0x03;
/// Not-equal
pub const _MM_CMPINT_NE: _MM_CMPINT_ENUM = 0x04;
/// Not less-than
pub const _MM_CMPINT_NLT: _MM_CMPINT_ENUM = 0x05;
/// Not less-than-or-equal
pub const _MM_CMPINT_NLE: _MM_CMPINT_ENUM = 0x06;
/// True
pub const _MM_CMPINT_TRUE: _MM_CMPINT_ENUM = 0x07;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.gather.dpd.512"]
    fn vgatherdpd(src: f64x8, slice: *const i8, offsets: i32x8, mask: i8, scale: i32) -> f64x8;
    #[link_name = "llvm.x86.avx512.gather.dps.512"]
    fn vgatherdps(src: f32x16, slice: *const i8, offsets: i32x16, mask: i16, scale: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.gather.qpd.512"]
    fn vgatherqpd(src: f64x8, slice: *const i8, offsets: i64x8, mask: i8, scale: i32) -> f64x8;
    #[link_name = "llvm.x86.avx512.gather.qps.512"]
    fn vgatherqps(src: f32x8, slice: *const i8, offsets: i64x8, mask: i8, scale: i32) -> f32x8;
    #[link_name = "llvm.x86.avx512.gather.dpq.512"]
    fn vpgatherdq(src: i64x8, slice: *const i8, offsets: i32x8, mask: i8, scale: i32) -> i64x8;
    #[link_name = "llvm.x86.avx512.gather.dpi.512"]
    fn vpgatherdd(src: i32x16, slice: *const i8, offsets: i32x16, mask: i16, scale: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.gather.qpq.512"]
    fn vpgatherqq(src: i64x8, slice: *const i8, offsets: i64x8, mask: i8, scale: i32) -> i64x8;
    #[link_name = "llvm.x86.avx512.gather.qpi.512"]
    fn vpgatherqd(src: i32x8, slice: *const i8, offsets: i64x8, mask: i8, scale: i32) -> i32x8;

    #[link_name = "llvm.x86.avx512.scatter.dpd.512"]
    fn vscatterdpd(slice: *mut i8, mask: i8, offsets: i32x8, src: f64x8, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.dps.512"]
    fn vscatterdps(slice: *mut i8, mask: i16, offsets: i32x16, src: f32x16, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.qpd.512"]
    fn vscatterqpd(slice: *mut i8, mask: i8, offsets: i64x8, src: f64x8, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.qps.512"]
    fn vscatterqps(slice: *mut i8, mask: i8, offsets: i64x8, src: f32x8, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.dpq.512"]
    fn vpscatterdq(slice: *mut i8, mask: i8, offsets: i32x8, src: i64x8, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.dpi.512"]
    fn vpscatterdd(slice: *mut i8, mask: i16, offsets: i32x16, src: i32x16, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.qpq.512"]
    fn vpscatterqq(slice: *mut i8, mask: i8, offsets: i64x8, src: i64x8, scale: i32);
    #[link_name = "llvm.x86.avx512.scatter.qpi.512"]
    fn vpscatterqd(slice: *mut i8, mask: i8, offsets: i64x8, src: i32x8, scale: i32);

    #[link_name = "llvm.x86.avx512.mask.cmp.ss"]
    fn vcmpss(a: __m128, b: __m128, op: i32, m: i8, sae: i32) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.sd"]
    fn vcmpsd(a: __m128d, b: __m128d, op: i32, m: i8, sae: i32) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.ps.512"]
    fn vcmpps(a: f32x16, b: f32x16, op: i32, m: i16, sae: i32) -> i16;
    #[link_name = "llvm.x86.avx512.mask.cmp.pd.512"]
    fn vcmppd(a: f64x8, b: f64x8, op: i32, m: i8, sae: i32) -> i8;
    #[link_name = "llvm.x86.avx512.mask.ucmp.q.512"]
    fn vpcmpuq(a: i64x8, b: i64x8, op: i32, m: i8) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.q.512"]
    fn vpcmpq(a: i64x8, b: i64x8, op: i32, m: i8) -> i8;
    #[link_name = "llvm.x86.avx512.mask.ucmp.d.512"]
    fn vpcmpud(a: i32x16, b: i32x16, op: i32, m: i16) -> i16;
    #[link_name = "llvm.x86.avx512.mask.cmp.d.512"]
    fn vpcmpd(a: i32x16, b: i32x16, op: i32, m: i16) -> i16;

    #[link_name = "llvm.x86.avx512.mask.prol.d.512"]
    fn vprold(a: i32x16, i8: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.pror.d.512"]
    fn vprord(a: i32x16, i8: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.prol.q.512"]
    fn vprolq(a: i64x8, i8: i32) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pror.q.512"]
    fn vprorq(a: i64x8, i8: i32) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.prolv.d.512"]
    fn vprolvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.prorv.d.512"]
    fn vprorvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.prolv.q.512"]
    fn vprolvq(a: i64x8, b: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.prorv.q.512"]
    fn vprorvq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.psllv.d.512"]
    fn vpsllvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrlv.d.512"]
    fn vpsrlvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.psllv.q.512"]
    fn vpsllvq(a: i64x8, b: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.psrlv.q.512"]
    fn vpsrlvq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.pslli.d.512"]
    fn vpsllid(a: i32x16, imm8: u32) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrli.d.512"]
    fn vpsrlid(a: i32x16, imm8: u32) -> i32x16;
    #[link_name = "llvm.x86.avx512.pslli.q.512"]
    fn vpslliq(a: i64x8, imm8: u32) -> i64x8;
    #[link_name = "llvm.x86.avx512.psrli.q.512"]
    fn vpsrliq(a: i64x8, imm8: u32) -> i64x8;

    #[link_name = "llvm.x86.avx512.psll.d.512"]
    fn vpslld(a: i32x16, count: i32x4) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrl.d.512"]
    fn vpsrld(a: i32x16, count: i32x4) -> i32x16;
    #[link_name = "llvm.x86.avx512.psll.q.512"]
    fn vpsllq(a: i64x8, count: i64x2) -> i64x8;
    #[link_name = "llvm.x86.avx512.psrl.q.512"]
    fn vpsrlq(a: i64x8, count: i64x2) -> i64x8;

    #[link_name = "llvm.x86.avx512.psra.d.512"]
    fn vpsrad(a: i32x16, count: i32x4) -> i32x16;
    #[link_name = "llvm.x86.avx512.psra.q.512"]
    fn vpsraq(a: i64x8, count: i64x2) -> i64x8;

    #[link_name = "llvm.x86.avx512.psrai.d.512"]
    fn vpsraid(a: i32x16, imm8: u32) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrai.q.512"]
    fn vpsraiq(a: i64x8, imm8: u32) -> i64x8;

    #[link_name = "llvm.x86.avx512.psrav.d.512"]
    fn vpsravd(a: i32x16, count: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrav.q.512"]
    fn vpsravq(a: i64x8, count: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.kand.w"]
    fn kandw(ma: u16, mb: u16) -> u16;
    #[link_name = "llvm.x86.avx512.kor.w"]
    fn korw(ma: u16, mb: u16) -> u16;
    #[link_name = "llvm.x86.avx512.kxor.w"]
    fn kxorw(ma: u16, mb: u16) -> u16;
}

#[cfg(test)]
mod tests {
    use std;
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::hint::black_box;

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm512_abs_epi32(a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm512_mask_abs_epi32(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi32(a, 0b11111111, a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm512_maskz_abs_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi32(0b11111111, a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32gather_ps() {
        let mut arr = [0f32; 256];
        for i in 0..256 {
            arr[i] = i as f32;
        }
        // A multiplier of 4 is word-addressing
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      120, 128, 136, 144, 152, 160, 168, 176);
        let r = _mm512_i32gather_ps(index, arr.as_ptr() as *const u8, 4);
        #[rustfmt::skip]
        assert_eq_m512(r, _mm512_setr_ps(0., 16., 32., 48., 64., 80., 96., 112.,
                                         120., 128., 136., 144., 152., 160., 168., 176.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32gather_ps() {
        let mut arr = [0f32; 256];
        for i in 0..256 {
            arr[i] = i as f32;
        }
        let src = _mm512_set1_ps(2.);
        let mask = 0b10101010_10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      120, 128, 136, 144, 152, 160, 168, 176);
        // A multiplier of 4 is word-addressing
        let r = _mm512_mask_i32gather_ps(src, mask, index, arr.as_ptr() as *const u8, 4);
        #[rustfmt::skip]
        assert_eq_m512(r, _mm512_setr_ps(2., 16., 2., 48., 2., 80., 2., 112.,
                                         2., 128., 2., 144., 2., 160., 2., 176.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32gather_epi32() {
        let mut arr = [0i32; 256];
        for i in 0..256 {
            arr[i] = i as i32;
        }
        // A multiplier of 4 is word-addressing
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      120, 128, 136, 144, 152, 160, 168, 176);
        let r = _mm512_i32gather_epi32(index, arr.as_ptr() as *const u8, 4);
        #[rustfmt::skip]
        assert_eq_m512i(r, _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                             120, 128, 136, 144, 152, 160, 168, 176));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32gather_epi32() {
        let mut arr = [0i32; 256];
        for i in 0..256 {
            arr[i] = i as i32;
        }
        let src = _mm512_set1_epi32(2);
        let mask = 0b10101010_10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      128, 144, 160, 176, 192, 208, 224, 240);
        // A multiplier of 4 is word-addressing
        let r = _mm512_mask_i32gather_epi32(src, mask, index, arr.as_ptr() as *const u8, 4);
        #[rustfmt::skip]
        assert_eq_m512i(r, _mm512_setr_epi32(2, 16, 2, 48, 2, 80, 2, 112,
                                             2, 144, 2, 176, 2, 208, 2, 240));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32scatter_ps() {
        let mut arr = [0f32; 256];
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      128, 144, 160, 176, 192, 208, 224, 240);
        let src = _mm512_setr_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        // A multiplier of 4 is word-addressing
        _mm512_i32scatter_ps(arr.as_mut_ptr() as *mut u8, index, src, 4);
        let mut expected = [0f32; 256];
        for i in 0..16 {
            expected[i * 16] = (i + 1) as f32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32scatter_ps() {
        let mut arr = [0f32; 256];
        let mask = 0b10101010_10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      128, 144, 160, 176, 192, 208, 224, 240);
        let src = _mm512_setr_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        // A multiplier of 4 is word-addressing
        _mm512_mask_i32scatter_ps(arr.as_mut_ptr() as *mut u8, mask, index, src, 4);
        let mut expected = [0f32; 256];
        for i in 0..8 {
            expected[i * 32 + 16] = 2. * (i + 1) as f32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32scatter_epi32() {
        let mut arr = [0i32; 256];
        #[rustfmt::skip]

        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      128, 144, 160, 176, 192, 208, 224, 240);
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        // A multiplier of 4 is word-addressing
        _mm512_i32scatter_epi32(arr.as_mut_ptr() as *mut u8, index, src, 4);
        let mut expected = [0i32; 256];
        for i in 0..16 {
            expected[i * 16] = (i + 1) as i32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32scatter_epi32() {
        let mut arr = [0i32; 256];
        let mask = 0b10101010_10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                      128, 144, 160, 176, 192, 208, 224, 240);
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        // A multiplier of 4 is word-addressing
        _mm512_mask_i32scatter_epi32(arr.as_mut_ptr() as *mut u8, mask, index, src, 4);
        let mut expected = [0i32; 256];
        for i in 0..8 {
            expected[i * 32 + 16] = 2 * (i + 1) as i32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmplt_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let m = _mm512_cmplt_ps_mask(a, b);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmplt_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmplt_ps_mask(mask, a, b);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpnlt_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        assert_eq!(_mm512_cmpnlt_ps_mask(a, b), !_mm512_cmplt_ps_mask(a, b));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpnlt_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let mask = 0b01111010_01111010;
        assert_eq!(_mm512_mask_cmpnlt_ps_mask(mask, a, b), 0b01111010_01111010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpnle_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let m = _mm512_cmpnle_ps_mask(b, a);
        assert_eq!(m, 0b00001101_00001101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpnle_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmpnle_ps_mask(mask, b, a);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        assert_eq!(_mm512_cmple_ps_mask(a, b), 0b00100101_00100101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.,
                              0., 1., -1., f32::MAX, f32::NAN, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let mask = 0b01111010_01111010;
        assert_eq!(_mm512_mask_cmple_ps_mask(mask, a, b), 0b00100000_00100000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.);
        let m = _mm512_cmpeq_ps_mask(b, a);
        assert_eq!(m, 0b11001101_11001101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpeq_ps_mask(mask, b, a);
        assert_eq!(r, 0b01001000_01001000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.);
        let m = _mm512_cmpneq_ps_mask(b, a);
        assert_eq!(m, 0b00110010_00110010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, f32::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.,
                              0., 1., 13., 42., f32::MAX, f32::MIN, f32::NAN, -100.);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpneq_ps_mask(mask, b, a);
        assert_eq!(r, 0b00110010_00110010)
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let m = _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmp_ps_mask(mask, a, b, _CMP_LT_OQ);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_round_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let m = _mm512_cmp_round_ps_mask(a, b, _CMP_LT_OQ, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_round_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.,
                              0., 1., -1., 13., f32::MAX, f32::MIN, 100., -100.);
        let b = _mm512_set1_ps(-1.);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmp_round_ps_mask(mask, a, b, _CMP_LT_OQ, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpord_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, -1., f32::NAN, 0.,
                              f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, 1., f32::NAN, 2.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 0.,
                              f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 2.);
        let m = _mm512_cmpord_ps_mask(a, b);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpord_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, -1., f32::NAN, 0.,
                              f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, 1., f32::NAN, 2.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 0.,
                              f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 2.);
        let mask = 0b11000011_11000011;
        let m = _mm512_mask_cmpord_ps_mask(mask, a, b);
        assert_eq!(m, 0b00000001_00000001);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpunord_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, -1., f32::NAN, 0.,
                              f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, 1., f32::NAN, 2.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 0.,
                              f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 2.);
        let m = _mm512_cmpunord_ps_mask(a, b);

        assert_eq!(m, 0b11111010_11111010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpunord_ps_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_ps(f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, -1., f32::NAN, 0.,
                              f32::NAN, f32::MAX, f32::NAN, f32::MIN, f32::NAN, 1., f32::NAN, 2.);
        #[rustfmt::skip]
        let b = _mm512_set_ps(f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 0.,
                              f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::MIN, f32::MAX, -1., 2.);
        let mask = 0b00001111_00001111;
        let m = _mm512_mask_cmpunord_ps_mask(mask, a, b);
        assert_eq!(m, 0b000001010_00001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_cmp_ss_mask() {
        let a = _mm_setr_ps(2., 1., 1., 1.);
        let b = _mm_setr_ps(1., 2., 2., 2.);
        let m = _mm_cmp_ss_mask(a, b, _CMP_GE_OS);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_mask_cmp_ss_mask() {
        let a = _mm_setr_ps(2., 1., 1., 1.);
        let b = _mm_setr_ps(1., 2., 2., 2.);
        let m = _mm_mask_cmp_ss_mask(0b10, a, b, _CMP_GE_OS);
        assert_eq!(m, 0);
        let m = _mm_mask_cmp_ss_mask(0b1, a, b, _CMP_GE_OS);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_cmp_round_ss_mask() {
        let a = _mm_setr_ps(2., 1., 1., 1.);
        let b = _mm_setr_ps(1., 2., 2., 2.);
        let m = _mm_cmp_round_ss_mask(a, b, _CMP_GE_OS, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_mask_cmp_round_ss_mask() {
        let a = _mm_setr_ps(2., 1., 1., 1.);
        let b = _mm_setr_ps(1., 2., 2., 2.);
        let m = _mm_mask_cmp_round_ss_mask(0b10, a, b, _CMP_GE_OS, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 0);
        let m = _mm_mask_cmp_round_ss_mask(0b1, a, b, _CMP_GE_OS, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_cmp_sd_mask() {
        let a = _mm_setr_pd(2., 1.);
        let b = _mm_setr_pd(1., 2.);
        let m = _mm_cmp_sd_mask(a, b, _CMP_GE_OS);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_mask_cmp_sd_mask() {
        let a = _mm_setr_pd(2., 1.);
        let b = _mm_setr_pd(1., 2.);
        let m = _mm_mask_cmp_sd_mask(0b10, a, b, _CMP_GE_OS);
        assert_eq!(m, 0);
        let m = _mm_mask_cmp_sd_mask(0b1, a, b, _CMP_GE_OS);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_cmp_round_sd_mask() {
        let a = _mm_setr_pd(2., 1.);
        let b = _mm_setr_pd(1., 2.);
        let m = _mm_cmp_round_sd_mask(a, b, _CMP_GE_OS, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm_mask_cmp_round_sd_mask() {
        let a = _mm_setr_pd(2., 1.);
        let b = _mm_setr_pd(1., 2.);
        let m = _mm_mask_cmp_round_sd_mask(0b10, a, b, _CMP_GE_OS, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 0);
        let m = _mm_mask_cmp_round_sd_mask(0b1, a, b, _CMP_GE_OS, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmplt_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmplt_epu32_mask(a, b);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmplt_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmplt_epu32_mask(mask, a, b);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpgt_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmpgt_epu32_mask(b, a);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpgt_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpgt_epu32_mask(mask, b, a);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmple_epu32_mask(a, b),
            !_mm512_cmpgt_epu32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(
            _mm512_mask_cmple_epu32_mask(mask, a, b),
            0b01111010_01111010
        );
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpge_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmpge_epu32_mask(a, b),
            !_mm512_cmplt_epu32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpge_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(_mm512_mask_cmpge_epu32_mask(mask, a, b), 0b01100000_0110000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpeq_epu32_mask(b, a);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpeq_epu32_mask(mask, b, a);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpneq_epu32_mask(b, a);
        assert_eq!(m, !_mm512_cmpeq_epu32_mask(b, a));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, -100, 100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, -100, 100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpneq_epu32_mask(mask, b, a);
        assert_eq!(r, 0b00110010_00110010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmp_epu32_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmplt_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmplt_epi32_mask(a, b);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmplt_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmplt_epi32_mask(mask, a, b);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpgt_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmpgt_epi32_mask(b, a);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpgt_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmpgt_epi32_mask(mask, b, a);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmple_epi32_mask(a, b),
            !_mm512_cmpgt_epi32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(_mm512_mask_cmple_epi32_mask(mask, a, b), 0b01100000_0110000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpge_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmpge_epi32_mask(a, b),
            !_mm512_cmplt_epi32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpge_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(
            _mm512_mask_cmpge_epi32_mask(mask, a, b),
            0b01111010_01111010
        );
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpeq_epi32_mask(b, a);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpeq_epi32_mask(mask, b, a);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpneq_epi32_mask(b, a);
        assert_eq!(m, !_mm512_cmpeq_epi32_mask(b, a));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, -100, 100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, -100, 100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpneq_epi32_mask(mask, b, a);
        assert_eq!(r, 0b00110010_00110010)
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b00000100_00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set_epi32() {
        let r = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(
            r,
            _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setr_epi32() {
        let r = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(
            r,
            _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set1_epi32() {
        let r = _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, _mm512_set1_epi32(2));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setzero_si512() {
        assert_eq_m512i(_mm512_set1_epi32(0), _mm512_setzero_si512());
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set_ps() {
        let r = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(
            r,
            _mm512_set_ps(
                15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
            ),
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setr_ps() {
        let r = _mm512_set_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(
            r,
            _mm512_setr_ps(
                15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
            ),
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set1_ps() {
        #[rustfmt::skip]
        let expected = _mm512_set_ps(2., 2., 2., 2., 2., 2., 2., 2.,
                                     2., 2., 2., 2., 2., 2., 2., 2.);
        assert_eq_m512(expected, _mm512_set1_ps(2.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setzero_ps() {
        assert_eq_m512(_mm512_setzero_ps(), _mm512_set1_ps(0.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_loadu_pd() {
        let a = &[4., 3., 2., 5., 8., 9., 64., 50.];
        let p = a.as_ptr();
        let r = _mm512_loadu_pd(black_box(p));
        let e = _mm512_setr_pd(4., 3., 2., 5., 8., 9., 64., 50.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_storeu_pd() {
        let a = _mm512_set1_pd(9.);
        let mut r = _mm512_undefined_pd();
        _mm512_storeu_pd(&mut r as *mut _ as *mut f64, a);
        assert_eq_m512d(r, a);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_loadu_ps() {
        let a = &[
            4., 3., 2., 5., 8., 9., 64., 50., -4., -3., -2., -5., -8., -9., -64., -50.,
        ];
        let p = a.as_ptr();
        let r = _mm512_loadu_ps(black_box(p));
        let e = _mm512_setr_ps(
            4., 3., 2., 5., 8., 9., 64., 50., -4., -3., -2., -5., -8., -9., -64., -50.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_storeu_ps() {
        let a = _mm512_set1_ps(9.);
        let mut r = _mm512_undefined_ps();
        _mm512_storeu_ps(&mut r as *mut _ as *mut f32, a);
        assert_eq_m512(r, a);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setr_pd() {
        let r = _mm512_set_pd(0., 1., 2., 3., 4., 5., 6., 7.);
        assert_eq_m512d(r, _mm512_setr_pd(7., 6., 5., 4., 3., 2., 1., 0.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set_pd() {
        let r = _mm512_setr_pd(0., 1., 2., 3., 4., 5., 6., 7.);
        assert_eq_m512d(r, _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_rol_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_rol_epi32(a, 1);
        let e = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_rol_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_mask_rol_epi32(a, 0, a, 1);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_rol_epi32(a, 0b11111111_11111111, a, 1);
        let e = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_rol_epi32() {
        let a = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        let r = _mm512_maskz_rol_epi32(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_rol_epi32(0b00000000_11111111, a, 1);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1 << 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_ror_epi32() {
        let a = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let r = _mm512_ror_epi32(a, 1);
        let e = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_ror_epi32() {
        let a = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let r = _mm512_mask_ror_epi32(a, 0, a, 1);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_ror_epi32(a, 0b11111111_11111111, a, 1);
        let e = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_ror_epi32() {
        let a = _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1 << 0);
        let r = _mm512_maskz_ror_epi32(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_ror_epi32(0b00000000_11111111, a, 1);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_slli_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_slli_epi32(a, 1);
        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_slli_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_mask_slli_epi32(a, 0, a, 1);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_slli_epi32(a, 0b11111111_11111111, a, 1);
        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_slli_epi32() {
        let a = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        let r = _mm512_maskz_slli_epi32(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_slli_epi32(0b00000000_11111111, a, 1);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srli_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let r = _mm512_srli_epi32(a, 1);
        let e = _mm512_set_epi32(0 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srli_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let r = _mm512_mask_srli_epi32(a, 0, a, 1);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srli_epi32(a, 0b11111111_11111111, a, 1);
        let e = _mm512_set_epi32(0 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srli_epi32() {
        let a = _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0);
        let r = _mm512_maskz_srli_epi32(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srli_epi32(0b00000000_11111111, a, 1);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0 << 31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_rolv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_rolv_epi32(a, b);

        let e = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_rolv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_mask_rolv_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_rolv_epi32(a, 0b11111111_11111111, a, b);

        let e = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_rolv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_maskz_rolv_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_rolv_epi32(0b00000000_11111111, a, b);

        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1 << 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_rorv_epi32() {
        let a = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_rorv_epi32(a, b);

        let e = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_rorv_epi32() {
        let a = _mm512_set_epi32(1 << 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_mask_rorv_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_rorv_epi32(a, 0b11111111_11111111, a, b);

        let e = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_rorv_epi32() {
        let a = _mm512_set_epi32(3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1 << 0);
        let b = _mm512_set_epi32(2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_maskz_rorv_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_rorv_epi32(0b00000000_11111111, a, b);

        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sllv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_sllv_epi32(a, count);

        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sllv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_mask_sllv_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_sllv_epi32(a, 0b11111111_11111111, a, count);

        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sllv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        let count = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_maskz_sllv_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_sllv_epi32(0b00000000_11111111, a, count);

        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srlv_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_srlv_epi32(a, count);

        let e = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srlv_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_mask_srlv_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srlv_epi32(a, 0b11111111_11111111, a, count);

        let e = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srlv_epi32() {
        let a = _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0);
        let count = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_maskz_srlv_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srlv_epi32(0b00000000_11111111, a, count);

        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sll_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_sll_epi32(a, count);
        let e = _mm512_set_epi32(
            0,
            1 << 2,
            1 << 3,
            1 << 4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sll_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_mask_sll_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_sll_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(
            0,
            1 << 2,
            1 << 3,
            1 << 4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sll_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 31,
        );
        let count = _mm_set_epi32(2, 0, 0, 2);
        let r = _mm512_maskz_sll_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_sll_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srl_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_srl_epi32(a, count);
        let e = _mm512_set_epi32(1 << 29, 0, 0, 1 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srl_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_mask_srl_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srl_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(1 << 29, 0, 0, 1 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srl_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 31,
        );
        let count = _mm_set_epi32(2, 0, 0, 2);
        let r = _mm512_maskz_srl_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srl_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 29);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sra_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        let count = _mm_set_epi32(1, 0, 0, 2);
        let r = _mm512_sra_epi32(a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sra_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16);
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_mask_sra_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_sra_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sra_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, -14);
        let count = _mm_set_epi32(2, 0, 0, 2);
        let r = _mm512_maskz_sra_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_sra_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srav_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        let count = _mm512_set_epi32(2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm512_srav_epi32(a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srav_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16);
        let count = _mm512_set_epi32(2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        let r = _mm512_mask_srav_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srav_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srav_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, -14);
        let count = _mm512_set_epi32(2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2);
        let r = _mm512_maskz_srav_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srav_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srai_epi32() {
        let a = _mm512_set_epi32(8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, -15);
        let r = _mm512_srai_epi32(a, 2);
        let e = _mm512_set_epi32(2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srai_epi32() {
        let a = _mm512_set_epi32(8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, -15);
        let r = _mm512_mask_srai_epi32(a, 0, a, 2);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srai_epi32(a, 0b11111111_11111111, a, 2);
        let e = _mm512_set_epi32(2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srai_epi32() {
        let a = _mm512_set_epi32(8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, -15);
        let r = _mm512_maskz_srai_epi32(0, a, 2);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srai_epi32(0b00000000_11111111, a, 2);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_and_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_and_epi32(a, b);
        let e = _mm512_set_epi32(1 << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_and_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_mask_and_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_and_epi32(a, 0b01111111_11111111, a, b);
        let e = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_and_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_maskz_and_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_and_epi32(0b00000000_11111111, a, b);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_and_si512() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_and_epi32(a, b);
        let e = _mm512_set_epi32(1 << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_or_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_or_epi32(a, b);
        let e = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_or_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_mask_or_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_or_epi32(a, 0b11111111_11111111, a, b);
        let e = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_or_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_maskz_or_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_or_epi32(0b00000000_11111111, a, b);
        let e = _mm512_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_or_si512() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_or_epi32(a, b);
        let e = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_xor_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_xor_epi32(a, b);
        let e = _mm512_set_epi32(
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_xor_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_mask_xor_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_xor_epi32(a, 0b01111111_11111111, a, b);
        let e = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_xor_epi32() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_maskz_xor_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_xor_epi32(0b00000000_11111111, a, b);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 1 | 1 << 4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_xor_si512() {
        let a = _mm512_set_epi32(
            1 << 1 | 1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 3,
        );
        let b = _mm512_set_epi32(
            1 << 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 3 | 1 << 4,
        );
        let r = _mm512_xor_epi32(a, b);
        let e = _mm512_set_epi32(
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 1 | 1 << 4,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kand() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b11001100_00110011;
        let r = _mm512_kand(a, b);
        let e: u16 = 0b11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kand_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b11001100_00110011;
        let r = _kand_mask16(a, b);
        let e: u16 = 0b11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kor() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kor(a, b);
        let e: u16 = 0b11101110_00111011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kor_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _kor_mask16(a, b);
        let e: u16 = 0b11101110_00111011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kxor() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kxor(a, b);
        let e: u16 = 0b11100010_00111000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kxor_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _kxor_mask16(a, b);
        let e: u16 = 0b11100010_00111000;
        assert_eq!(r, e);
    }
}

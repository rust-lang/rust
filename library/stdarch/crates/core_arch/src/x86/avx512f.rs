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

/// Finds the absolute value of each packed single-precision (32-bit) floating-point element in v2, storing the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_abs_ps&expand=65)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_abs_ps(v2: __m512) -> __m512 {
    let a = _mm512_set1_epi32(0x7FFFFFFF); // from LLVM code
    let b = transmute::<f32x16, __m512i>(v2.as_f32x16());
    let abs = _mm512_and_epi32(a, b);
    transmute(abs)
}

/// Finds the absolute value of each packed single-precision (32-bit) floating-point element in v2, storing the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_abs_ps&expand=66)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandd))]
pub unsafe fn _mm512_mask_abs_ps(src: __m512, k: __mmask16, v2: __m512) -> __m512 {
    let abs = _mm512_abs_ps(v2).as_f32x16();
    transmute(simd_select_bitmask(k, abs, src.as_f32x16()))
}

/// Finds the absolute value of each packed double-precision (64-bit) floating-point element in v2, storing the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_abs_pd&expand=60)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_abs_pd(v2: __m512d) -> __m512d {
    let a = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF); // from LLVM code
    let b = transmute::<f64x8, __m512i>(v2.as_f64x8());
    let abs = _mm512_and_epi64(a, b);
    transmute(abs)
}

/// Finds the absolute value of each packed double-precision (64-bit) floating-point element in v2, storing the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_abs_pd&expand=61)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_mask_abs_pd(src: __m512d, k: __mmask8, v2: __m512d) -> __m512d {
    let abs = _mm512_abs_pd(v2).as_f64x8();
    transmute(simd_select_bitmask(k, abs, src.as_f64x8()))
}

/// Add packed 32-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_epi32&expand=100)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i32x16(), b.as_i32x16()))
}

/// Add packed 32-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_epi32&expand=101)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_mask_add_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, add, src.as_i32x16()))
}

/// Add packed 32-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_epi32&expand=102)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_maskz_add_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 64-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_epi64&expand=109)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_add_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i64x8(), b.as_i64x8()))
}

/// Add packed 64-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_epi64&expand=110)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_mask_add_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, add, src.as_i64x8()))
}

/// Add packed 64-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_epi64&expand=111)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_maskz_add_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_ps&expand=139)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_add_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_add(a.as_f32x16(), b.as_f32x16()))
}

/// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_ps&expand=140)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_mask_add_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let add = _mm512_add_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, add, src.as_f32x16()))
}

/// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_ps&expand=141)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_maskz_add_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let add = _mm512_add_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_pd&expand=127)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_add_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_add(a.as_f64x8(), b.as_f64x8()))
}

/// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_pd&expand=128)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_mask_add_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let add = _mm512_add_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, add, src.as_f64x8()))
}

/// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_pd&expand=129)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_maskz_add_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let add = _mm512_add_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_epi32&expand=5694)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm512_sub_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i32x16(), b.as_i32x16()))
}

/// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_epi32&expand=5692)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm512_mask_sub_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, sub, src.as_i32x16()))
}

/// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_epi32&expand=5693)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm512_maskz_sub_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_epi64&expand=5703)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm512_sub_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i64x8(), b.as_i64x8()))
}

/// Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_epi64&expand=5701)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm512_mask_sub_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, sub, src.as_i64x8()))
}

/// Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_epi64&expand=5702)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm512_maskz_sub_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_sub_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_ps&expand=5733)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubps))]
pub unsafe fn _mm512_sub_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_sub(a.as_f32x16(), b.as_f32x16()))
}

/// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_ps&expand=5731)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubps))]
pub unsafe fn _mm512_mask_sub_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let sub = _mm512_sub_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, sub, src.as_f32x16()))
}

/// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_ps&expand=5732)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubps))]
pub unsafe fn _mm512_maskz_sub_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let sub = _mm512_sub_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_pd&expand=5721)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubpd))]
pub unsafe fn _mm512_sub_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_sub(a.as_f64x8(), b.as_f64x8()))
}

/// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_pd&expand=5719)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubpd))]
pub unsafe fn _mm512_mask_sub_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let sub = _mm512_sub_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, sub, src.as_f64x8()))
}

/// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_pd&expand=5720)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubpd))]
pub unsafe fn _mm512_maskz_sub_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let sub = _mm512_sub_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Multiply the low signed 32-bit integers from each packed 64-bit element in a and b, and store the signed 64-bit results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mul_epi32&expand=3907)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm512_mul_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmuldq(a.as_i32x16(), b.as_i32x16()))
}

/// Multiply the low signed 32-bit integers from each packed 64-bit element in a and b, and store the signed 64-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_epi32&expand=3905)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm512_mask_mul_epi32(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mul_epi32(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, mul, src.as_i64x8()))
}

/// Multiply the low signed 32-bit integers from each packed 64-bit element in a and b, and store the signed 64-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_epi32&expand=3906)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm512_maskz_mul_epi32(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mul_epi32(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers, and store the low 32 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mullo_epi&expand=4005)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm512_mullo_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_mul(a.as_i32x16(), b.as_i32x16()))
}

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers, and store the low 32 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mullo_epi32&expand=4003)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm512_mask_mullo_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mullo_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, mul, src.as_i32x16()))
}

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers, and store the low 32 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mullo_epi32&expand=4004)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm512_maskz_mullo_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mullo_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiplies elements in packed 64-bit integer vectors a and b together, storing the lower 64 bits of the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mullox_epi64&expand=4017)
///
/// This intrinsic generates a sequence of instructions, which may perform worse than a native instruction. Consider the performance impact of this intrinsic.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_mullox_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_mul(a.as_i64x8(), b.as_i64x8()))
}

/// Multiplies elements in packed 64-bit integer vectors a and b together, storing the lower 64 bits of the result in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mullox&expand=4016)
///
/// This intrinsic generates a sequence of instructions, which may perform worse than a native instruction. Consider the performance impact of this intrinsic.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_mask_mullox_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mullox_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, mul, src.as_i64x8()))
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element in a and b, and store the unsigned 64-bit results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mul_epu32&expand=3916)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuludq))]
pub unsafe fn _mm512_mul_epu32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmuludq(a.as_u32x16(), b.as_u32x16()))
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element in a and b, and store the unsigned 64-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_epu32&expand=3914)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuludq))]
pub unsafe fn _mm512_mask_mul_epu32(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mul_epu32(a, b).as_u64x8();
    transmute(simd_select_bitmask(k, mul, src.as_u64x8()))
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element in a and b, and store the unsigned 64-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_epu32&expand=3915)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuludq))]
pub unsafe fn _mm512_maskz_mul_epu32(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mul_epu32(a, b).as_u64x8();
    let zero = _mm512_setzero_si512().as_u64x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mm512_mul_ps&expand=3934)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulps))]
pub unsafe fn _mm512_mul_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_mul(a.as_f32x16(), b.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). RM.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_ps&expand=3932)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulps))]
pub unsafe fn _mm512_mask_mul_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let mul = _mm512_mul_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, mul, src.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_ps&expand=3933)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulps))]
pub unsafe fn _mm512_maskz_mul_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let mul = _mm512_mul_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mul_pd&expand=3925)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulpd))]
pub unsafe fn _mm512_mul_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_mul(a.as_f64x8(), b.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). RM.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_pd&expand=3923)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulpd))]
pub unsafe fn _mm512_mask_mul_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let mul = _mm512_mul_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, mul, src.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_pd&expand=3924)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulpd))]
pub unsafe fn _mm512_maskz_mul_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let mul = _mm512_mul_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_div_ps&expand=2162)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivps))]
pub unsafe fn _mm512_div_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_div(a.as_f32x16(), b.as_f32x16()))
}

/// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_div_ps&expand=2163)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivps))]
pub unsafe fn _mm512_mask_div_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let div = _mm512_div_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, div, src.as_f32x16()))
}

/// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_div_ps&expand=2164)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivps))]
pub unsafe fn _mm512_maskz_div_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let div = _mm512_div_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, div, zero))
}

/// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_div_pd&expand=2153)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivpd))]
pub unsafe fn _mm512_div_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_div(a.as_f64x8(), b.as_f64x8()))
}

/// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_div_pd&expand=2154)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivpd))]
pub unsafe fn _mm512_mask_div_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let div = _mm512_div_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, div, src.as_f64x8()))
}

/// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_div_pd&expand=2155)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivpd))]
pub unsafe fn _mm512_maskz_div_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let div = _mm512_div_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, div, zero))
}

/// Compare packed signed 32-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_epi32&expand=3582)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm512_max_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsd(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_epi32&expand=3580)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm512_mask_max_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, max, src.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_epi32&expand=3581)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm512_maskz_max_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 64-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_epi64&expand=3591)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsq))]
pub unsafe fn _mm512_max_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_epi64&expand=3589)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsq))]
pub unsafe fn _mm512_mask_max_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, max, src.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_epi64&expand=3590)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsq))]
pub unsafe fn _mm512_maskz_max_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_ps&expand=3655)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_max_ps(a: __m512, b: __m512) -> __m512 {
    transmute(vmaxps(
        a.as_f32x16(),
        b.as_f32x16(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_ps&expand=3653)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_mask_max_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let max = _mm512_max_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, max, src.as_f32x16()))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_ps&expand=3654)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_maskz_max_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let max = _mm512_max_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_pd&expand=3645)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_max_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(vmaxpd(a.as_f64x8(), b.as_f64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_pd&expand=3643)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_mask_max_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let max = _mm512_max_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, max, src.as_f64x8()))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_pd&expand=3644)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_maskz_max_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let max = _mm512_max_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 32-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_epu32&expand=3618)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxud))]
pub unsafe fn _mm512_max_epu32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxud(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_epu32&expand=3616)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxud))]
pub unsafe fn _mm512_mask_max_epu32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu32(a, b).as_u32x16();
    transmute(simd_select_bitmask(k, max, src.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_epu32&expand=3617)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxud))]
pub unsafe fn _mm512_maskz_max_epu32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu32(a, b).as_u32x16();
    let zero = _mm512_setzero_si512().as_u32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 64-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=max_epu64&expand=3627)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxuq))]
pub unsafe fn _mm512_max_epu64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxuq(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_epu64&expand=3625)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxuq))]
pub unsafe fn _mm512_mask_max_epu64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu64(a, b).as_u64x8();
    transmute(simd_select_bitmask(k, max, src.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_epu&expand=3626)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxuq))]
pub unsafe fn _mm512_maskz_max_epu64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu64(a, b).as_u64x8();
    let zero = _mm512_setzero_si512().as_u64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 32-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_epi32&expand=3696)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm512_min_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminsd(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_epi32&expand=3694)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm512_mask_min_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, max, src.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_epi32&expand=3695)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm512_maskz_min_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 64-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_epi64&expand=3705)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsq))]
pub unsafe fn _mm512_min_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminsq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_epi64&expand=3703)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsq))]
pub unsafe fn _mm512_mask_min_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, max, src.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_epi64&expand=3704)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsq))]
pub unsafe fn _mm512_maskz_min_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_ps&expand=3769)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_min_ps(a: __m512, b: __m512) -> __m512 {
    transmute(vminps(
        a.as_f32x16(),
        b.as_f32x16(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_ps&expand=3767)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_mask_min_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let max = _mm512_min_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, max, src.as_f32x16()))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_ps&expand=3768)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_maskz_min_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let max = _mm512_min_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_pd&expand=3759)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_min_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(vminpd(a.as_f64x8(), b.as_f64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_pd&expand=3757)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_mask_min_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let max = _mm512_min_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, max, src.as_f64x8()))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_pd&expand=3758)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_maskz_min_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let max = _mm512_min_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 32-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_epu32&expand=3732)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminud))]
pub unsafe fn _mm512_min_epu32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminud(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_epu32&expand=3730)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminud))]
pub unsafe fn _mm512_mask_min_epu32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epu32(a, b).as_u32x16();
    transmute(simd_select_bitmask(k, max, src.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_epu32&expand=3731)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminud))]
pub unsafe fn _mm512_maskz_min_epu32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epu32(a, b).as_u32x16();
    let zero = _mm512_setzero_si512().as_u32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 64-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_epu64&expand=3741)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminuq))]
pub unsafe fn _mm512_min_epu64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminuq(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_epu64&expand=3739)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminuq))]
pub unsafe fn _mm512_mask_min_epu64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epu64(a, b).as_u64x8();
    transmute(simd_select_bitmask(k, max, src.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_epu64&expand=3740)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminuq))]
pub unsafe fn _mm512_maskz_min_epu64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epu64(a, b).as_u64x8();
    let zero = _mm512_setzero_si512().as_u64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sqrt_ps&expand=5371)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtps))]
pub unsafe fn _mm512_sqrt_ps(a: __m512) -> __m512 {
    transmute(vsqrtps(a.as_f32x16(), _MM_FROUND_CUR_DIRECTION))
}

/// Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sqrt_ps&expand=5369)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtps))]
pub unsafe fn _mm512_mask_sqrt_ps(src: __m512, k: __mmask16, a: __m512) -> __m512 {
    let sqrt = _mm512_sqrt_ps(a).as_f32x16();
    transmute(simd_select_bitmask(k, sqrt, src.as_f32x16()))
}

/// Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sqrt_ps&expand=5370)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtps))]
pub unsafe fn _mm512_maskz_sqrt_ps(k: __mmask16, a: __m512) -> __m512 {
    let sqrt = _mm512_sqrt_ps(a).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, sqrt, zero))
}

/// Compute the square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sqrt_pd&expand=5362)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtpd))]
pub unsafe fn _mm512_sqrt_pd(a: __m512d) -> __m512d {
    transmute(vsqrtpd(a.as_f64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Compute the square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sqrt_pd&expand=5360)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtpd))]
pub unsafe fn _mm512_mask_sqrt_pd(src: __m512d, k: __mmask8, a: __m512d) -> __m512d {
    let sqrt = _mm512_sqrt_pd(a).as_f64x8();
    transmute(simd_select_bitmask(k, sqrt, src.as_f64x8()))
}

/// Compute the square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sqrt_pd&expand=5361)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtpd))]
pub unsafe fn _mm512_maskz_sqrt_pd(k: __mmask8, a: __m512d) -> __m512d {
    let sqrt = _mm512_sqrt_pd(a).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, sqrt, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=fmadd_ps&expand=2557)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
pub unsafe fn _mm512_fmadd_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    transmute(vfmadd132ps(
        a.as_f32x16(),
        b.as_f32x16(),
        c.as_f32x16(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmadd_ps&expand=2558)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
pub unsafe fn _mm512_mask_fmadd_ps(a: __m512, k: __mmask16, b: __m512, c: __m512) -> __m512 {
    let fmadd = _mm512_fmadd_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmadd, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmadd_ps&expand=2560)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
pub unsafe fn _mm512_maskz_fmadd_ps(k: __mmask16, a: __m512, b: __m512, c: __m512) -> __m512 {
    let fmadd = _mm512_fmadd_ps(a, b, c).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fmadd, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmadd_ps&expand=2559)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
pub unsafe fn _mm512_mask3_fmadd_ps(a: __m512, b: __m512, c: __m512, k: __mmask16) -> __m512 {
    let fmadd = _mm512_fmadd_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmadd, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmadd_pd&expand=2545)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
pub unsafe fn _mm512_fmadd_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    transmute(vfmadd132pd(
        a.as_f64x8(),
        b.as_f64x8(),
        c.as_f64x8(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmadd_pd&expand=2546)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
pub unsafe fn _mm512_mask_fmadd_pd(a: __m512d, k: __mmask8, b: __m512d, c: __m512d) -> __m512d {
    let fmadd = _mm512_fmadd_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmadd, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmadd_pd&expand=2548)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
pub unsafe fn _mm512_maskz_fmadd_pd(k: __mmask8, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let fmadd = _mm512_fmadd_pd(a, b, c).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fmadd, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmadd_pd&expand=2547)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
pub unsafe fn _mm512_mask3_fmadd_pd(a: __m512d, b: __m512d, c: __m512d, k: __mmask8) -> __m512d {
    let fmadd = _mm512_fmadd_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmadd, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsub_ps&expand=2643)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generate vfmadd, gcc generate vfmsub
pub unsafe fn _mm512_fmsub_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    transmute(vfmadd132ps(
        a.as_f32x16(),
        b.as_f32x16(),
        sub,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsub_ps&expand=2644)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generate vfmadd, gcc generate vfmsub
pub unsafe fn _mm512_mask_fmsub_ps(a: __m512, k: __mmask16, b: __m512, c: __m512) -> __m512 {
    let fmsub = _mm512_fmsub_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmsub, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsub_ps&expand=2646)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generate vfmadd, gcc generate vfmsub
pub unsafe fn _mm512_maskz_fmsub_ps(k: __mmask16, a: __m512, b: __m512, c: __m512) -> __m512 {
    let fmsub = _mm512_fmsub_ps(a, b, c).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fmsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsub_ps&expand=2645)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generate vfmadd, gcc generate vfmsub
pub unsafe fn _mm512_mask3_fmsub_ps(a: __m512, b: __m512, c: __m512, k: __mmask16) -> __m512 {
    let fmsub = _mm512_fmsub_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmsub, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsub_pd&expand=2631)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang fmadd, gcc fmsub
pub unsafe fn _mm512_fmsub_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    transmute(vfmadd132pd(
        a.as_f64x8(),
        b.as_f64x8(),
        sub,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsub_pd&expand=2632)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang fmadd, gcc fmsub
pub unsafe fn _mm512_mask_fmsub_pd(a: __m512d, k: __mmask8, b: __m512d, c: __m512d) -> __m512d {
    let fmsub = _mm512_fmsub_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmsub, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsub_pd&expand=2634)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang fmadd, gcc fmsub
pub unsafe fn _mm512_maskz_fmsub_pd(k: __mmask8, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let fmsub = _mm512_fmsub_pd(a, b, c).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fmsub, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsub_pd&expand=2633)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang fmadd, gcc fmsub
pub unsafe fn _mm512_mask3_fmsub_pd(a: __m512d, b: __m512d, c: __m512d, k: __mmask8) -> __m512d {
    let fmsub = _mm512_fmsub_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmsub, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmaddsub_ps&expand=2611)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
pub unsafe fn _mm512_fmaddsub_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    transmute(vfmaddsub213ps(
        a.as_f32x16(),
        b.as_f32x16(),
        c.as_f32x16(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmaddsub_ps&expand=2612)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
pub unsafe fn _mm512_mask_fmaddsub_ps(a: __m512, k: __mmask16, b: __m512, c: __m512) -> __m512 {
    let fmaddsub = _mm512_fmaddsub_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmaddsub, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmaddsub_ps&expand=2614)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
pub unsafe fn _mm512_maskz_fmaddsub_ps(k: __mmask16, a: __m512, b: __m512, c: __m512) -> __m512 {
    let fmaddsub = _mm512_fmaddsub_ps(a, b, c).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fmaddsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmaddsub_ps&expand=2613)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
pub unsafe fn _mm512_mask3_fmaddsub_ps(a: __m512, b: __m512, c: __m512, k: __mmask16) -> __m512 {
    let fmaddsub = _mm512_fmaddsub_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmaddsub, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmaddsub_pd&expand=2599)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
pub unsafe fn _mm512_fmaddsub_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    transmute(vfmaddsub213pd(
        a.as_f64x8(),
        b.as_f64x8(),
        c.as_f64x8(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmaddsub_pd&expand=2600)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
pub unsafe fn _mm512_mask_fmaddsub_pd(a: __m512d, k: __mmask8, b: __m512d, c: __m512d) -> __m512d {
    let fmaddsub = _mm512_fmaddsub_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmaddsub, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmaddsub_pd&expand=2602)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
pub unsafe fn _mm512_maskz_fmaddsub_pd(k: __mmask8, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let fmaddsub = _mm512_fmaddsub_pd(a, b, c).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fmaddsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmaddsub_ps&expand=2613)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
pub unsafe fn _mm512_mask3_fmaddsub_pd(a: __m512d, b: __m512d, c: __m512d, k: __mmask8) -> __m512d {
    let fmaddsub = _mm512_fmaddsub_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmaddsub, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsubadd_ps&expand=2691)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
pub unsafe fn _mm512_fmsubadd_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    transmute(vfmaddsub213ps(
        a.as_f32x16(),
        b.as_f32x16(),
        sub,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsubadd_ps&expand=2692)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
pub unsafe fn _mm512_mask_fmsubadd_ps(a: __m512, k: __mmask16, b: __m512, c: __m512) -> __m512 {
    let fmsubadd = _mm512_fmsubadd_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmsubadd, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsubadd_ps&expand=2694)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
pub unsafe fn _mm512_maskz_fmsubadd_ps(k: __mmask16, a: __m512, b: __m512, c: __m512) -> __m512 {
    let fmsubadd = _mm512_fmsubadd_ps(a, b, c).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fmsubadd, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsubadd_ps&expand=2693)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
pub unsafe fn _mm512_mask3_fmsubadd_ps(a: __m512, b: __m512, c: __m512, k: __mmask16) -> __m512 {
    let fmsubadd = _mm512_fmsubadd_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fmsubadd, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsubadd_pd&expand=2679)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
pub unsafe fn _mm512_fmsubadd_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    transmute(vfmaddsub213pd(
        a.as_f64x8(),
        b.as_f64x8(),
        sub,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsubadd_pd&expand=2680)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
pub unsafe fn _mm512_mask_fmsubadd_pd(a: __m512d, k: __mmask8, b: __m512d, c: __m512d) -> __m512d {
    let fmsubadd = _mm512_fmsubadd_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmsubadd, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsubadd_pd&expand=2682)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
pub unsafe fn _mm512_maskz_fmsubadd_pd(k: __mmask8, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let fmsubadd = _mm512_fmsubadd_pd(a, b, c).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fmsubadd, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsubadd_pd&expand=2681)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
pub unsafe fn _mm512_mask3_fmsubadd_pd(a: __m512d, b: __m512d, c: __m512d, k: __mmask8) -> __m512d {
    let fmsubadd = _mm512_fmsubadd_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fmsubadd, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmadd_ps&expand=2723)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
pub unsafe fn _mm512_fnmadd_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f32x16());
    transmute(vfmadd132ps(
        sub,
        b.as_f32x16(),
        c.as_f32x16(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmadd_ps&expand=2724)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
pub unsafe fn _mm512_mask_fnmadd_ps(a: __m512, k: __mmask16, b: __m512, c: __m512) -> __m512 {
    let fnmadd = _mm512_fnmadd_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fnmadd, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmadd_ps&expand=2726)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
pub unsafe fn _mm512_maskz_fnmadd_ps(k: __mmask16, a: __m512, b: __m512, c: __m512) -> __m512 {
    let fnmadd = _mm512_fnmadd_ps(a, b, c).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fnmadd, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmadd_ps&expand=2725)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
pub unsafe fn _mm512_mask3_fnmadd_ps(a: __m512, b: __m512, c: __m512, k: __mmask16) -> __m512 {
    let fnmadd = _mm512_fnmadd_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fnmadd, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmadd_pd&expand=2711)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
pub unsafe fn _mm512_fnmadd_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f64x8());
    transmute(vfmadd132pd(
        sub,
        b.as_f64x8(),
        c.as_f64x8(),
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmadd_pd&expand=2712)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
pub unsafe fn _mm512_mask_fnmadd_pd(a: __m512d, k: __mmask8, b: __m512d, c: __m512d) -> __m512d {
    let fnmadd = _mm512_fnmadd_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fnmadd, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmadd_pd&expand=2714)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
pub unsafe fn _mm512_maskz_fnmadd_pd(k: __mmask8, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let fnmadd = _mm512_fnmadd_pd(a, b, c).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fnmadd, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmadd_pd&expand=2713)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
pub unsafe fn _mm512_mask3_fnmadd_pd(a: __m512d, b: __m512d, c: __m512d, k: __mmask8) -> __m512d {
    let fnmadd = _mm512_fnmadd_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fnmadd, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmsub_ps&expand=2771)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
pub unsafe fn _mm512_fnmsub_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f32x16());
    let subc = simd_sub(zero, c.as_f32x16());
    transmute(vfmadd132ps(
        suba,
        b.as_f32x16(),
        subc,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmsub_ps&expand=2772)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
pub unsafe fn _mm512_mask_fnmsub_ps(a: __m512, k: __mmask16, b: __m512, c: __m512) -> __m512 {
    let fnmsub = _mm512_fnmsub_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fnmsub, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmsub_ps&expand=2774)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
pub unsafe fn _mm512_maskz_fnmsub_ps(k: __mmask16, a: __m512, b: __m512, c: __m512) -> __m512 {
    let fnmsub = _mm512_fnmsub_ps(a, b, c).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fnmsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmsub_ps&expand=2773)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
pub unsafe fn _mm512_mask3_fnmsub_ps(a: __m512, b: __m512, c: __m512, k: __mmask16) -> __m512 {
    let fnmsub = _mm512_fnmsub_ps(a, b, c).as_f32x16();
    transmute(simd_select_bitmask(k, fnmsub, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmsub_pd&expand=2759)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
pub unsafe fn _mm512_fnmsub_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f64x8());
    let subc = simd_sub(zero, c.as_f64x8());
    transmute(vfmadd132pd(
        suba,
        b.as_f64x8(),
        subc,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmsub_pd&expand=2760)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
pub unsafe fn _mm512_mask_fnmsub_pd(a: __m512d, k: __mmask8, b: __m512d, c: __m512d) -> __m512d {
    let fnmsub = _mm512_fnmsub_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fnmsub, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmsub_pd&expand=2762)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
pub unsafe fn _mm512_maskz_fnmsub_pd(k: __mmask8, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let fnmsub = _mm512_fnmsub_pd(a, b, c).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fnmsub, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmsub_pd&expand=2761)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
pub unsafe fn _mm512_mask3_fnmsub_pd(a: __m512d, b: __m512d, c: __m512d, k: __mmask8) -> __m512d {
    let fnmsub = _mm512_fnmsub_pd(a, b, c).as_f64x8();
    transmute(simd_select_bitmask(k, fnmsub, c.as_f64x8()))
}

/// Compute the approximate reciprocal of packed single-precision (32-bit) floating-point elements in a, and store the results in dst. The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rcp14_ps&expand=4502)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrcp14ps))]
pub unsafe fn _mm512_rcp14_ps(a: __m512) -> __m512 {
    transmute(vrcp14ps(
        a.as_f32x16(),
        _mm512_setzero_ps().as_f32x16(),
        0b11111111_11111111,
    ))
}

/// Compute the approximate reciprocal of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rcp14_ps&expand=4500)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrcp14ps))]
pub unsafe fn _mm512_mask_rcp14_ps(src: __m512, k: __mmask16, a: __m512) -> __m512 {
    transmute(vrcp14ps(a.as_f32x16(), src.as_f32x16(), k))
}

/// Compute the approximate reciprocal of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rcp14_ps&expand=4501)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrcp14ps))]
pub unsafe fn _mm512_maskz_rcp14_ps(k: __mmask16, a: __m512) -> __m512 {
    transmute(vrcp14ps(a.as_f32x16(), _mm512_setzero_ps().as_f32x16(), k))
}

/// Compute the approximate reciprocal of packed double-precision (64-bit) floating-point elements in a, and store the results in dst. The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rcp14_pd&expand=4493)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrcp14pd))]
pub unsafe fn _mm512_rcp14_pd(a: __m512d) -> __m512d {
    transmute(vrcp14pd(
        a.as_f64x8(),
        _mm512_setzero_pd().as_f64x8(),
        0b11111111,
    ))
}

/// Compute the approximate reciprocal of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rcp14_pd&expand=4491)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrcp14pd))]
pub unsafe fn _mm512_mask_rcp14_pd(src: __m512d, k: __mmask8, a: __m512d) -> __m512d {
    transmute(vrcp14pd(a.as_f64x8(), src.as_f64x8(), k))
}

/// Compute the approximate reciprocal of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rcp14_pd&expand=4492)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrcp14pd))]
pub unsafe fn _mm512_maskz_rcp14_pd(k: __mmask8, a: __m512d) -> __m512d {
    transmute(vrcp14pd(a.as_f64x8(), _mm512_setzero_pd().as_f64x8(), k))
}

/// Compute the approximate reciprocal square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst. The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rsqrt14_ps&expand=4819)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrsqrt14ps))]
pub unsafe fn _mm512_rsqrt14_ps(a: __m512) -> __m512 {
    transmute(vrsqrt14ps(
        a.as_f32x16(),
        _mm512_setzero_ps().as_f32x16(),
        0b11111111_11111111,
    ))
}

/// Compute the approximate reciprocal square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rsqrt14_ps&expand=4817)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrsqrt14ps))]
pub unsafe fn _mm512_mask_rsqrt14_ps(src: __m512, k: __mmask16, a: __m512) -> __m512 {
    transmute(vrsqrt14ps(a.as_f32x16(), src.as_f32x16(), k))
}

/// Compute the approximate reciprocal square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rsqrt14_ps&expand=4818)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrsqrt14ps))]
pub unsafe fn _mm512_maskz_rsqrt14_ps(k: __mmask16, a: __m512) -> __m512 {
    transmute(vrsqrt14ps(
        a.as_f32x16(),
        _mm512_setzero_ps().as_f32x16(),
        k,
    ))
}

/// Compute the approximate reciprocal square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst. The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_rsqrt14_pd&expand=4812)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrsqrt14pd))]
pub unsafe fn _mm512_rsqrt14_pd(a: __m512d) -> __m512d {
    transmute(vrsqrt14pd(
        a.as_f64x8(),
        _mm512_setzero_pd().as_f64x8(),
        0b11111111,
    ))
}

/// Compute the approximate reciprocal square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rsqrt14_pd&expand=4810)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrsqrt14pd))]
pub unsafe fn _mm512_mask_rsqrt14_pd(src: __m512d, k: __mmask8, a: __m512d) -> __m512d {
    transmute(vrsqrt14pd(a.as_f64x8(), src.as_f64x8(), k))
}

/// Compute the approximate reciprocal square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). The maximum relative error for this approximation is less than 2^-14.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_rsqrt14_pd&expand=4811)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vrsqrt14pd))]
pub unsafe fn _mm512_maskz_rsqrt14_pd(k: __mmask8, a: __m512d) -> __m512d {
    transmute(vrsqrt14pd(a.as_f64x8(), _mm512_setzero_pd().as_f64x8(), k))
}

/// Convert the exponent of each packed single-precision (32-bit) floating-point element in a to a single-precision (32-bit) floating-point number representing the integer exponent, and store the results in dst. This intrinsic essentially calculates floor(log2(x)) for each element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getexp_ps&expand=2844)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexpps))]
pub unsafe fn _mm512_getexp_ps(a: __m512) -> __m512 {
    transmute(vgetexpps(
        a.as_f32x16(),
        _mm512_setzero_ps().as_f32x16(),
        0b11111111_11111111,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert the exponent of each packed single-precision (32-bit) floating-point element in a to a single-precision (32-bit) floating-point number representing the integer exponent, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getexp_ps&expand=2845)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexpps))]
pub unsafe fn _mm512_mask_getexp_ps(src: __m512, k: __mmask16, a: __m512) -> __m512 {
    transmute(vgetexpps(
        a.as_f32x16(),
        src.as_f32x16(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert the exponent of each packed single-precision (32-bit) floating-point element in a to a single-precision (32-bit) floating-point number representing the integer exponent, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getexp_ps&expand=2846)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexpps))]
pub unsafe fn _mm512_maskz_getexp_ps(k: __mmask16, a: __m512) -> __m512 {
    transmute(vgetexpps(
        a.as_f32x16(),
        _mm512_setzero_ps().as_f32x16(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert the exponent of each packed double-precision (64-bit) floating-point element in a to a double-precision (64-bit) floating-point number representing the integer exponent, and store the results in dst. This intrinsic essentially calculates floor(log2(x)) for each element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getexp_pd&expand=2835)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexppd))]
pub unsafe fn _mm512_getexp_pd(a: __m512d) -> __m512d {
    transmute(vgetexppd(
        a.as_f64x8(),
        _mm512_setzero_pd().as_f64x8(),
        0b11111111,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert the exponent of each packed double-precision (64-bit) floating-point element in a to a double-precision (64-bit) floating-point number representing the integer exponent, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getexp_pd&expand=2836)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexppd))]
pub unsafe fn _mm512_mask_getexp_pd(src: __m512d, k: __mmask8, a: __m512d) -> __m512d {
    transmute(vgetexppd(
        a.as_f64x8(),
        src.as_f64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert the exponent of each packed double-precision (64-bit) floating-point element in a to a double-precision (64-bit) floating-point number representing the integer exponent, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getexp_pd&expand=2837)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexppd))]
pub unsafe fn _mm512_maskz_getexp_pd(k: __mmask8, a: __m512d) -> __m512d {
    transmute(vgetexppd(
        a.as_f64x8(),
        _mm512_setzero_pd().as_f64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Normalize the mantissas of packed single-precision (32-bit) floating-point elements in a, and store the results in dst. This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getmant_ps&expand=2880)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantps, norm = 0, sign = 0))]
#[rustc_args_required_const(1, 2)]
pub unsafe fn _mm512_getmant_ps(
    a: __m512,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr, $imm2:expr) => {
            vgetmantps(
                a.as_f32x16(),
                $imm2 << 2 | $imm4,
                _mm512_setzero_ps().as_f32x16(),
                0b11111111_11111111,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm4_mantissas!(norm, sign, call);
    transmute(r)
}

/// Normalize the mantissas of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getmant_ps&expand=2881)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantps, norm = 0, sign = 0))]
#[rustc_args_required_const(3, 4)]
pub unsafe fn _mm512_mask_getmant_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr, $imm2:expr) => {
            vgetmantps(
                a.as_f32x16(),
                $imm2 << 2 | $imm4,
                src.as_f32x16(),
                k,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm4_mantissas!(norm, sign, call);
    transmute(r)
}

/// Normalize the mantissas of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getmant_ps&expand=2882)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantps, norm = 0, sign = 0))]
#[rustc_args_required_const(2, 3)]
pub unsafe fn _mm512_maskz_getmant_ps(
    k: __mmask16,
    a: __m512,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr, $imm2:expr) => {
            vgetmantps(
                a.as_f32x16(),
                $imm2 << 2 | $imm4,
                _mm512_setzero_ps().as_f32x16(),
                k,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm4_mantissas!(norm, sign, call);
    transmute(r)
}

/// Normalize the mantissas of packed double-precision (64-bit) floating-point elements in a, and store the results in dst. This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getmant_pd&expand=2871)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantpd, norm = 0, sign = 0))]
#[rustc_args_required_const(1, 2)]
pub unsafe fn _mm512_getmant_pd(
    a: __m512d,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr, $imm2:expr) => {
            vgetmantpd(
                a.as_f64x8(),
                $imm2 << 2 | $imm4,
                _mm512_setzero_pd().as_f64x8(),
                0b11111111,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm4_mantissas!(norm, sign, call);
    transmute(r)
}

/// Normalize the mantissas of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getmant_pd&expand=2872)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantpd, norm = 0, sign = 0))]
#[rustc_args_required_const(3, 4)]
pub unsafe fn _mm512_mask_getmant_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr, $imm2:expr) => {
            vgetmantpd(
                a.as_f64x8(),
                $imm2 << 2 | $imm4,
                src.as_f64x8(),
                k,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm4_mantissas!(norm, sign, call);
    transmute(r)
}

/// Normalize the mantissas of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getmant_pd&expand=2873)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantpd, norm = 0, sign = 0))]
#[rustc_args_required_const(2, 3)]
pub unsafe fn _mm512_maskz_getmant_pd(
    k: __mmask8,
    a: __m512d,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr, $imm2:expr) => {
            vgetmantpd(
                a.as_f64x8(),
                $imm2 << 2 | $imm4,
                _mm512_setzero_pd().as_f64x8(),
                k,
                _MM_FROUND_CUR_DIRECTION,
            )
        };
    }
    let r = constify_imm4_mantissas!(norm, sign, call);
    transmute(r)
}

/// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_round_ps&expand=145)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_add_round_ps(a: __m512, b: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vaddps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_round_ps&expand=146)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_add_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vaddps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let addround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, addround, src.as_f32x16()))
}

/// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_round_ps&expand=147)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_add_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vaddps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let addround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, addround, zero))
}

/// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_round_pd&expand=142)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_add_round_pd(a: __m512d, b: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vaddpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_round_pd&expand=143)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_add_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vaddpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let addround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, addround, src.as_f64x8()))
}

/// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_round_pd&expand=144)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_add_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vaddpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let addround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, addround, zero))
}

/// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_round_ps&expand=5739)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubps, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_sub_round_ps(a: __m512, b: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vsubps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_round_ps&expand=5737)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubps, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_sub_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vsubps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let subround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, subround, src.as_f32x16()))
}

/// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_round_ps&expand=5738)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubps, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_sub_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vsubps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let subround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, subround, zero))
}

/// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_round_pd&expand=5736)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubpd, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_sub_round_pd(a: __m512d, b: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vsubpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_round_pd&expand=5734)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubpd, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_sub_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vsubpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let subround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, subround, src.as_f64x8()))
}

/// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_round_pd&expand=5735)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsubpd, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_sub_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vsubpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let subround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, subround, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mul_round_ps&expand=3940)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulps, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_mul_round_ps(a: __m512, b: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vmulps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_round_ps&expand=3938)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulps, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_mul_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vmulps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let mulround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, mulround, src.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_round_ps&expand=3939)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulps, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_mul_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vmulps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let mulround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, mulround, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mul_round_pd&expand=3937)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulpd, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_mul_round_pd(a: __m512d, b: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vmulpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_round_pd&expand=3935)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulpd, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_mul_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vmulpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let mulround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, mulround, src.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_round_ps&expand=3939)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmulpd, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_mul_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vmulpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let mulround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, mulround, zero))
}

/// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_div_round_ps&expand=2168)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivps, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_div_round_ps(a: __m512, b: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vdivps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_div_round_ps&expand=2169)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivps, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_div_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vdivps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let divround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, divround, src.as_f32x16()))
}

/// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_div_round_ps&expand=2170)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivps, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_div_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vdivps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let divround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, divround, zero))
}

/// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, =and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_div_round_pd&expand=2165)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivpd, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_div_round_pd(a: __m512d, b: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vdivpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_div_round_pd&expand=2166)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivpd, rounding = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_div_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vdivpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let divround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, divround, src.as_f64x8()))
}

/// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_div_round_pd&expand=2167)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vdivpd, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_div_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vdivpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let divround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, divround, zero))
}

/// Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sqrt_round_ps&expand=5377)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtps, rounding = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_sqrt_round_ps(a: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vsqrtps(a.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sqrt_round_ps&expand=5375)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtps, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_sqrt_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vsqrtps(a.as_f32x16(), $imm4)
        };
    }
    let sqrtround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, sqrtround, src.as_f32x16()))
}

/// Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sqrt_round_ps&expand=5376)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtps, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_sqrt_round_ps(k: __mmask16, a: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vsqrtps(a.as_f32x16(), $imm4)
        };
    }
    let sqrtround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, sqrtround, zero))
}

/// Compute the square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sqrt_round_pd&expand=5374)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtpd, rounding = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_sqrt_round_pd(a: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vsqrtpd(a.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Compute the square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sqrt_round_pd&expand=5372)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtpd, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_sqrt_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vsqrtpd(a.as_f64x8(), $imm4)
        };
    }
    let sqrtround = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, sqrtround, src.as_f64x8()))
}

/// Compute the square root of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sqrt_round_pd&expand=5373)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vsqrtpd, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_sqrt_round_pd(k: __mmask8, a: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vsqrtpd(a.as_f64x8(), $imm4)
        };
    }
    let sqrtround = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, sqrtround, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmadd_round_ps&expand=2565)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmadd_round_ps(a: __m512, b: __m512, c: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmadd_round_ps&expand=2566)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmadd_round_ps(
    a: __m512,
    k: __mmask16,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmadd, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in a using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmadd_round_ps&expand=2568)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmadd_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fmadd = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fmadd, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmadd_round_ps&expand=2567)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132ps or vfmadd213ps or vfmadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmadd_round_ps(
    a: __m512,
    b: __m512,
    c: __m512,
    k: __mmask16,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmadd, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmadd_round_pd&expand=2561)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmadd_round_pd(a: __m512d, b: __m512d, c: __m512d, rounding: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmadd_round_pd&expand=2562)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmadd_round_pd(
    a: __m512d,
    k: __mmask8,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmadd, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmadd_round_pd&expand=2564)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmadd_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fmadd = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fmadd, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmadd_round_pd&expand=2563)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmadd132pd or vfmadd213pd or vfmadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmadd_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    k: __mmask8,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmadd, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsub_round_ps&expand=2651)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generates vfmadd, gcc generates vfmsub
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmsub_round_ps(a: __m512, b: __m512, c: __m512, rounding: i32) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsub_round_ps&expand=2652)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generates vfmadd, gcc generates vfmsub
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmsub_round_ps(
    a: __m512,
    k: __mmask16,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let fmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsub, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsub_round_ps&expand=2654)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generates vfmadd, gcc generates vfmsub
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmsub_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let fmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsub_round_ps&expand=2653)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132ps or vfmsub213ps or vfmsub231ps, clang generates vfmadd, gcc generates vfmsub
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmsub_round_ps(
    a: __m512,
    b: __m512,
    c: __m512,
    k: __mmask16,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let fmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsub, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsub_round_pd&expand=2647)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang generates fmadd, gcc generates fmsub
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmsub_round_pd(a: __m512d, b: __m512d, c: __m512d, rounding: i32) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsub_round_pd&expand=2648)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang generates fmadd, gcc generates fmsub
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmsub_round_pd(
    a: __m512d,
    k: __mmask8,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let fmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsub, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsub_round_pd&expand=2650)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang generates fmadd, gcc generates fmsub
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmsub_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let fmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsub, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsub_round_pd&expand=2649)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfmsub132pd or vfmsub213pd or vfmsub231pd. clang generates fmadd, gcc generates fmsub
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmsub_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    k: __mmask8,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let fmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsub, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmaddsub_round_ps&expand=2619)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmaddsub_round_ps(a: __m512, b: __m512, c: __m512, rounding: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmaddsub_round_ps&expand=2620)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmaddsub_round_ps(
    a: __m512,
    k: __mmask16,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fmaddsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmaddsub, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmaddsub_round_ps&expand=2622)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmaddsub_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fmaddsub = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, fmaddsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmaddsub_round_ps&expand=2621)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132ps or vfmaddsub213ps or vfmaddsub231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmaddsub_round_ps(
    a: __m512,
    b: __m512,
    c: __m512,
    k: __mmask16,
    rounding: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fmaddsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmaddsub, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmaddsub_round_pd&expand=2615)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmaddsub_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmaddsub_round_pd&expand=2616)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmaddsub_round_pd(
    a: __m512d,
    k: __mmask8,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fmaddsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmaddsub, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmaddsub_round_pd&expand=2618)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmaddsub_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fmaddsub = constify_imm4_round!(rounding, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, fmaddsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmaddsub_round_pd&expand=2617)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmaddsub132pd or vfmaddsub213pd or vfmaddsub231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmaddsub_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    k: __mmask8,
    rounding: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fmaddsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmaddsub, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsubadd_round_ps&expand=2699)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmsubadd_round_ps(a: __m512, b: __m512, c: __m512, rounding: i32) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsubadd_round_ps&expand=2700)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmsubadd_round_ps(
    a: __m512,
    k: __mmask16,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let fmsubadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsubadd, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsubadd_round_ps&expand=2702)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmsubadd_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let fmsubadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsubadd, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsubadd_round_ps&expand=2701)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132ps or vfmsubadd213ps or vfmsubadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmsubadd_round_ps(
    a: __m512,
    b: __m512,
    c: __m512,
    k: __mmask16,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213ps(a.as_f32x16(), b.as_f32x16(), sub, $imm4)
        };
    }
    let fmsubadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsubadd, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmsubadd_round_pd&expand=2695)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fmsubadd_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fmsubadd_round_pd&expand=2696)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fmsubadd_round_pd(
    a: __m512d,
    k: __mmask8,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let fmsubadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsubadd, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively add and subtract packed elements in c to/from the intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fmsubadd_round_pd&expand=2698)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fmsubadd_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let fmsubadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsubadd, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, alternatively subtract and add packed elements in c from/to the intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fmsubadd_round_pd&expand=2697)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmaddsub, rounding = 8))] //vfmsubadd132pd or vfmsubadd213pd or vfmsubadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fmsubadd_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    k: __mmask8,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmaddsub213pd(a.as_f64x8(), b.as_f64x8(), sub, $imm4)
        };
    }
    let fmsubadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fmsubadd, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmadd_round_ps&expand=2731)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fnmadd_round_ps(a: __m512, b: __m512, c: __m512, rounding: i32) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(sub, b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmadd_round_ps&expand=2732)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fnmadd_round_ps(
    a: __m512,
    k: __mmask16,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(sub, b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fnmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmadd, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmadd_round_ps&expand=2734)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fnmadd_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(sub, b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fnmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmadd, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmadd_round_ps&expand=2733)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132ps or vfnmadd213ps or vfnmadd231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fnmadd_round_ps(
    a: __m512,
    b: __m512,
    c: __m512,
    k: __mmask16,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(sub, b.as_f32x16(), c.as_f32x16(), $imm4)
        };
    }
    let fnmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmadd, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmadd_pd&expand=2711)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fnmadd_round_pd(a: __m512d, b: __m512d, c: __m512d, rounding: i32) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(sub, b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmadd_round_pd&expand=2728)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fnmadd_round_pd(
    a: __m512d,
    k: __mmask8,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(sub, b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fnmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmadd, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmadd_round_pd&expand=2730)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fnmadd_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(sub, b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fnmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmadd, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, add the negated intermediate result to packed elements in c, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmadd_round_pd&expand=2729)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmadd132pd or vfnmadd213pd or vfnmadd231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fnmadd_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    k: __mmask8,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let sub = simd_sub(zero, a.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(sub, b.as_f64x8(), c.as_f64x8(), $imm4)
        };
    }
    let fnmadd = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmadd, c.as_f64x8()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmsub_round_ps&expand=2779)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fnmsub_round_ps(a: __m512, b: __m512, c: __m512, rounding: i32) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f32x16());
    let subc = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(suba, b.as_f32x16(), subc, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmsub_round_ps&expand=2780)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fnmsub_round_ps(
    a: __m512,
    k: __mmask16,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f32x16());
    let subc = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(suba, b.as_f32x16(), subc, $imm4)
        };
    }
    let fnmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmsub, a.as_f32x16()))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmsub_round_ps&expand=2782)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fnmsub_round_ps(
    k: __mmask16,
    a: __m512,
    b: __m512,
    c: __m512,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f32x16());
    let subc = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(suba, b.as_f32x16(), subc, $imm4)
        };
    }
    let fnmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmsub, zero))
}

/// Multiply packed single-precision (32-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmsub_round_ps&expand=2781)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132ps or vfnmsub213ps or vfnmsub231ps
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fnmsub_round_ps(
    a: __m512,
    b: __m512,
    c: __m512,
    k: __mmask16,
    rounding: i32,
) -> __m512 {
    let zero: f32x16 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f32x16());
    let subc = simd_sub(zero, c.as_f32x16());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132ps(suba, b.as_f32x16(), subc, $imm4)
        };
    }
    let fnmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmsub, c.as_f32x16()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fnmsub_round_pd&expand=2775)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_fnmsub_round_pd(a: __m512d, b: __m512d, c: __m512d, rounding: i32) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f64x8());
    let subc = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(suba, b.as_f64x8(), subc, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_fnmsub_round_pd&expand=2776)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_fnmsub_round_pd(
    a: __m512d,
    k: __mmask8,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f64x8());
    let subc = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(suba, b.as_f64x8(), subc, $imm4)
        };
    }
    let fnmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmsub, a.as_f64x8()))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_fnmsub_round_pd&expand=2778)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_maskz_fnmsub_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    c: __m512d,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f64x8());
    let subc = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(suba, b.as_f64x8(), subc, $imm4)
        };
    }
    let fnmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmsub, zero))
}

/// Multiply packed double-precision (64-bit) floating-point elements in a and b, subtract packed elements in c from the negated intermediate result, and store the results in dst using writemask k (elements are copied from c when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask3_fnmsub_round_pd&expand=2777)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vfmadd, rounding = 8))] //vfnmsub132pd or vfnmsub213pd or vfnmsub231pd
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask3_fnmsub_round_pd(
    a: __m512d,
    b: __m512d,
    c: __m512d,
    k: __mmask8,
    rounding: i32,
) -> __m512d {
    let zero: f64x8 = mem::zeroed();
    let suba = simd_sub(zero, a.as_f64x8());
    let subc = simd_sub(zero, c.as_f64x8());
    macro_rules! call {
        ($imm4:expr) => {
            vfmadd132pd(suba, b.as_f64x8(), subc, $imm4)
        };
    }
    let fnmsub = constify_imm4_round!(rounding, call);
    transmute(simd_select_bitmask(k, fnmsub, c.as_f64x8()))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed maximum values in dst.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=max_round_ps&expand=3662)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_max_round_ps(a: __m512, b: __m512, sae: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vmaxps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_round_ps&expand=3660)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps, sae = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_max_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    sae: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vmaxps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    transmute(simd_select_bitmask(k, max, src.as_f32x16()))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_round_ps&expand=3661)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_max_round_ps(k: __mmask16, a: __m512, b: __m512, sae: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vmaxps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed maximum values in dst.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_round_pd&expand=3659)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_max_round_pd(a: __m512d, b: __m512d, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vmaxpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_round_pd&expand=3657)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd, sae = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_max_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vmaxpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    transmute(simd_select_bitmask(k, max, src.as_f64x8()))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_round_pd&expand=3658)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_max_round_pd(k: __mmask8, a: __m512d, b: __m512d, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vmaxpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed minimum values in dst.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_round_ps&expand=3776)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_min_round_ps(a: __m512, b: __m512, sae: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vminps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_round_ps&expand=3774)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps, sae = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_min_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    sae: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vminps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    transmute(simd_select_bitmask(k, max, src.as_f32x16()))
}

/// Compare packed single-precision (32-bit) floating-point elements in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_round_ps&expand=3775)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_min_round_ps(k: __mmask16, a: __m512, b: __m512, sae: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vminps(a.as_f32x16(), b.as_f32x16(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed minimum values in dst.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_round_pd&expand=3773)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_min_round_pd(a: __m512d, b: __m512d, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vminpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_round_pd&expand=3771)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd, sae = 8))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_min_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vminpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    transmute(simd_select_bitmask(k, max, src.as_f64x8()))
}

/// Compare packed double-precision (64-bit) floating-point elements in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_round_pd&expand=3772)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_min_round_pd(k: __mmask8, a: __m512d, b: __m512d, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vminpd(a.as_f64x8(), b.as_f64x8(), $imm4)
        };
    }
    let max = constify_imm4_sae!(sae, call);
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Convert the exponent of each packed single-precision (32-bit) floating-point element in a to a single-precision (32-bit) floating-point number representing the integer exponent, and store the results in dst. This intrinsic essentially calculates floor(log2(x)) for each element.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getexp_round_ps&expand=2850)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexpps, sae = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_getexp_round_ps(a: __m512, sae: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vgetexpps(
                a.as_f32x16(),
                _mm512_setzero_ps().as_f32x16(),
                0b11111111_11111111,
                $imm4,
            )
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert the exponent of each packed single-precision (32-bit) floating-point element in a to a single-precision (32-bit) floating-point number representing the integer exponent, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getexp_round_ps&expand=2851)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexpps, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_getexp_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    sae: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vgetexpps(a.as_f32x16(), src.as_f32x16(), k, $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert the exponent of each packed single-precision (32-bit) floating-point element in a to a single-precision (32-bit) floating-point number representing the integer exponent, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getexp_round_ps&expand=2852)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexpps, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_getexp_round_ps(k: __mmask16, a: __m512, sae: i32) -> __m512 {
    macro_rules! call {
        ($imm4:expr) => {
            vgetexpps(a.as_f32x16(), _mm512_setzero_ps().as_f32x16(), k, $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert the exponent of each packed double-precision (64-bit) floating-point element in a to a double-precision (64-bit) floating-point number representing the integer exponent, and store the results in dst. This intrinsic essentially calculates floor(log2(x)) for each element.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getexp_round_pd&expand=2847)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexppd, sae = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_getexp_round_pd(a: __m512d, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vgetexppd(
                a.as_f64x8(),
                _mm512_setzero_pd().as_f64x8(),
                0b11111111,
                $imm4,
            )
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert the exponent of each packed double-precision (64-bit) floating-point element in a to a double-precision (64-bit) floating-point number representing the integer exponent, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getexp_round_pd&expand=2848)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexppd, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_getexp_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vgetexppd(a.as_f64x8(), src.as_f64x8(), k, $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert the exponent of each packed double-precision (64-bit) floating-point element in a to a double-precision (64-bit) floating-point number representing the integer exponent, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates floor(log2(x)) for each element.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getexp_round_pd&expand=2849)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetexppd, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_getexp_round_pd(k: __mmask8, a: __m512d, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vgetexppd(a.as_f64x8(), _mm512_setzero_pd().as_f64x8(), k, $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Normalize the mantissas of packed single-precision (32-bit) floating-point elements in a, and store the results in dst. This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getmant_round_ps&expand=2886)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantps, norm = 0, sign = 0, sae = 4))]
#[rustc_args_required_const(1, 2, 3)]
pub unsafe fn _mm512_getmant_round_ps(
    a: __m512,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
    sae: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4_1:expr, $imm2:expr, $imm4_2:expr) => {
            vgetmantps(
                a.as_f32x16(),
                $imm2 << 2 | $imm4_1,
                _mm512_setzero_ps().as_f32x16(),
                0b11111111_11111111,
                $imm4_2,
            )
        };
    }
    let r = constify_imm4_mantissas_sae!(norm, sign, sae, call);
    transmute(r)
}

/// Normalize the mantissas of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getmant_round_ps&expand=2887)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantps, norm = 0, sign = 0, sae = 4))]
#[rustc_args_required_const(3, 4, 5)]
pub unsafe fn _mm512_mask_getmant_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
    sae: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4_1:expr, $imm2:expr, $imm4_2:expr) => {
            vgetmantps(
                a.as_f32x16(),
                $imm2 << 2 | $imm4_1,
                src.as_f32x16(),
                k,
                $imm4_2,
            )
        };
    }
    let r = constify_imm4_mantissas_sae!(norm, sign, sae, call);
    transmute(r)
}

/// Normalize the mantissas of packed single-precision (32-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getmant_round_ps&expand=2888)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantps, norm = 0, sign = 0, sae = 4))]
#[rustc_args_required_const(2, 3, 4)]
pub unsafe fn _mm512_maskz_getmant_round_ps(
    k: __mmask16,
    a: __m512,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
    sae: i32,
) -> __m512 {
    macro_rules! call {
        ($imm4_1:expr, $imm2:expr, $imm4_2:expr) => {
            vgetmantps(
                a.as_f32x16(),
                $imm2 << 2 | $imm4_1,
                _mm512_setzero_ps().as_f32x16(),
                k,
                $imm4_2,
            )
        };
    }
    let r = constify_imm4_mantissas_sae!(norm, sign, sae, call);
    transmute(r)
}

/// Normalize the mantissas of packed double-precision (64-bit) floating-point elements in a, and store the results in dst. This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_getmant_round_pd&expand=2883)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantpd, norm = 0, sign = 0, sae = 4))]
#[rustc_args_required_const(1, 2, 3)]
pub unsafe fn _mm512_getmant_round_pd(
    a: __m512d,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4_1:expr, $imm2:expr, $imm4_2:expr) => {
            vgetmantpd(
                a.as_f64x8(),
                $imm2 << 2 | $imm4_1,
                _mm512_setzero_pd().as_f64x8(),
                0b11111111,
                $imm4_2,
            )
        };
    }
    let r = constify_imm4_mantissas_sae!(norm, sign, sae, call);
    transmute(r)
}

/// Normalize the mantissas of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_getmant_round_pd&expand=2884)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantpd, norm = 0, sign = 0, sae = 4))]
#[rustc_args_required_const(3, 4, 5)]
pub unsafe fn _mm512_mask_getmant_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4_1:expr, $imm2:expr, $imm4_2:expr) => {
            vgetmantpd(
                a.as_f64x8(),
                $imm2 << 2 | $imm4_1,
                src.as_f64x8(),
                k,
                $imm4_2,
            )
        };
    }
    let r = constify_imm4_mantissas_sae!(norm, sign, sae, call);
    transmute(r)
}

/// Normalize the mantissas of packed double-precision (64-bit) floating-point elements in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). This intrinsic essentially calculates ±(2^k)*|x.significand|, where k depends on the interval range defined by interv and the sign depends on sc and the source sign.
/// The mantissa is normalized to the interval specified by interv, which can take the following values:
///    _MM_MANT_NORM_1_2     // interval [1, 2)
///    _MM_MANT_NORM_p5_2    // interval [0.5, 2)
///    _MM_MANT_NORM_p5_1    // interval [0.5, 1)
///    _MM_MANT_NORM_p75_1p5 // interval [0.75, 1.5)
/// The sign is determined by sc which can take the following values:
///    _MM_MANT_SIGN_src     // sign = sign(src)
///    _MM_MANT_SIGN_zero    // sign = 0
///    _MM_MANT_SIGN_nan     // dst = NaN if sign(src) = 1
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_getmant_round_pd&expand=2885)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vgetmantpd, norm = 0, sign = 0, sae = 4))]
#[rustc_args_required_const(2, 3, 4)]
pub unsafe fn _mm512_maskz_getmant_round_pd(
    k: __mmask8,
    a: __m512d,
    norm: _MM_MANTISSA_NORM_ENUM,
    sign: _MM_MANTISSA_SIGN_ENUM,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4_1:expr, $imm2:expr, $imm4_2:expr) => {
            vgetmantpd(
                a.as_f64x8(),
                $imm2 << 2 | $imm4_1,
                _mm512_setzero_pd().as_f64x8(),
                k,
                $imm4_2,
            )
        };
    }
    let r = constify_imm4_mantissas_sae!(norm, sign, sae, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=cvtps_epi32&expand=1737)   
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2dq))]
pub unsafe fn _mm512_cvtps_epi32(a: __m512) -> __m512i {
    transmute(vcvtps2dq(
        a.as_f32x16(),
        _mm512_setzero_si512().as_i32x16(),
        0b11111111_11111111,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtps_epi32&expand=1738)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2dq))]
pub unsafe fn _mm512_mask_cvtps_epi32(src: __m512i, k: __mmask16, a: __m512) -> __m512i {
    transmute(vcvtps2dq(
        a.as_f32x16(),
        src.as_i32x16(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtps_epi32&expand=1739)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2dq))]
pub unsafe fn _mm512_maskz_cvtps_epi32(k: __mmask16, a: __m512) -> __m512i {
    transmute(vcvtps2dq(
        a.as_f32x16(),
        _mm512_setzero_si512().as_i32x16(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 32-bit integers, and store the results in dst.
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtps_epu32&expand=1755)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2udq))]
pub unsafe fn _mm512_cvtps_epu32(a: __m512) -> __m512i {
    transmute(vcvtps2udq(
        a.as_f32x16(),
        _mm512_setzero_si512().as_u32x16(),
        0b11111111_11111111,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 32-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtps_epu32&expand=1756)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2udq))]
pub unsafe fn _mm512_mask_cvtps_epu32(src: __m512i, k: __mmask16, a: __m512) -> __m512i {
    transmute(vcvtps2udq(
        a.as_f32x16(),
        src.as_u32x16(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 32-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=maskz_cvt_roundps_epu32&expand=1343)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2udq))]
pub unsafe fn _mm512_maskz_cvtps_epu32(k: __mmask16, a: __m512) -> __m512i {
    transmute(vcvtps2udq(
        a.as_f32x16(),
        _mm512_setzero_si512().as_u32x16(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed double-precision (64-bit) floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtps_pd&expand=1769)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2pd))]
pub unsafe fn _mm512_cvtps_pd(a: __m256) -> __m512d {
    transmute(vcvtps2pd(
        a.as_f32x8(),
        _mm512_setzero_pd().as_f64x8(),
        0b11111111,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed double-precision (64-bit) floating-point elements, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtps_pd&expand=1770)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2pd))]
pub unsafe fn _mm512_mask_cvtps_pd(src: __m512d, k: __mmask8, a: __m256) -> __m512d {
    transmute(vcvtps2pd(
        a.as_f32x8(),
        src.as_f64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed double-precision (64-bit) floating-point elements, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtps_pd&expand=1771)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2pd))]
pub unsafe fn _mm512_maskz_cvtps_pd(k: __mmask8, a: __m256) -> __m512d {
    transmute(vcvtps2pd(
        a.as_f32x8(),
        _mm512_setzero_pd().as_f64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvt_roundps_epi32&expand=1335)   
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2dq, rounding = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_cvt_roundps_epi32(a: __m512, rounding: i32) -> __m512i {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2dq(
                a.as_f32x16(),
                _mm512_setzero_si512().as_i32x16(),
                0b11111111_11111111,
                $imm4,
            )
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvt_roundps_epi32&expand=1336)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2dq, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_cvt_roundps_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512,
    rounding: i32,
) -> __m512i {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2dq(a.as_f32x16(), src.as_i32x16(), k, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed 32-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvt_roundps_epi32&expand=1337)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2dq, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_cvt_roundps_epi32(k: __mmask16, a: __m512, rounding: i32) -> __m512i {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2dq(a.as_f32x16(), _mm512_setzero_si512().as_i32x16(), k, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 32-bit integers, and store the results in dst.
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvt_roundps_epu32&expand=1341)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2udq, rounding = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_cvt_roundps_epu32(a: __m512, rounding: i32) -> __m512i {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2udq(
                a.as_f32x16(),
                _mm512_setzero_si512().as_u32x16(),
                0b11111111_11111111,
                $imm4,
            )
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 32-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvt_roundps_epu32&expand=1342)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2udq, rounding = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_cvt_roundps_epu32(
    src: __m512i,
    k: __mmask16,
    a: __m512,
    rounding: i32,
) -> __m512i {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2udq(a.as_f32x16(), src.as_u32x16(), k, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 32-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// Rounding is done according to the rounding[3:0] parameter, which can be one of:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///    _MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=maskz_cvt_roundps_epu32&expand=1343)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2udq, rounding = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_cvt_roundps_epu32(k: __mmask16, a: __m512, rounding: i32) -> __m512i {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2udq(a.as_f32x16(), _mm512_setzero_si512().as_u32x16(), k, $imm4)
        };
    }
    let r = constify_imm4_round!(rounding, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed double-precision (64-bit) floating-point elements, and store the results in dst.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///    
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=cvt_roundps_pd&expand=1347)   
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2pd, sae = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_cvt_roundps_pd(a: __m256, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2pd(
                a.as_f32x8(),
                _mm512_setzero_pd().as_f64x8(),
                0b11111111,
                $imm4,
            )
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed double-precision (64-bit) floating-point elements, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvt_roundps_epi32&expand=1336)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2pd, sae = 8))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_cvt_roundps_pd(
    src: __m512d,
    k: __mmask8,
    a: __m256,
    sae: i32,
) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2pd(a.as_f32x8(), src.as_f64x8(), k, $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed double-precision (64-bit) floating-point elements, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvt_roundps_epi32&expand=1337)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vcvtps2pd, sae = 8))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_cvt_roundps_pd(k: __mmask8, a: __m256, sae: i32) -> __m512d {
    macro_rules! call {
        ($imm4:expr) => {
            vcvtps2pd(a.as_f32x8(), _mm512_setzero_pd().as_f64x8(), k, $imm4)
        };
    }
    let r = constify_imm4_sae!(sae, call);
    transmute(r)
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
    macro_rules! call {
        ($imm8:expr) => {
            vprold(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Rotate the bits in each packed 32-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rol_epi32&expand=4683)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_rol_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprold(a.as_i32x16(), $imm8)
        };
    }
    let rol = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vprold(a.as_i32x16(), $imm8)
        };
    }
    let rol = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, rol, zero))
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_ror_epi32&expand=4721)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_ror_epi32(a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprord(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_ror_epi32&expand=4719)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprold, imm8 = 123))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_ror_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprord(a.as_i32x16(), $imm8)
        };
    }
    let ror = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vprord(a.as_i32x16(), $imm8)
        };
    }
    let ror = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vprolq(a.as_i64x8(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Rotate the bits in each packed 64-bit integer in a to the left by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_rol_epi64&expand=4692)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_rol_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprolq(a.as_i64x8(), $imm8)
        };
    }
    let rol = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vprolq(a.as_i64x8(), $imm8)
        };
    }
    let rol = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vprorq(a.as_i64x8(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Rotate the bits in each packed 64-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_ror_epi64&expand=4728)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprolq, imm8 = 15))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_ror_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprorq(a.as_i64x8(), $imm8)
        };
    }
    let ror = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vprorq(a.as_i64x8(), $imm8)
        };
    }
    let ror = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsllid(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_slli_epi32&expand=5308)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_slli_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsllid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsllid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsrlid(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srli_epi32&expand=5520)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srli_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsrlid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsrlid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpslliq(a.as_i64x8(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 64-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_slli_epi64&expand=5317)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllq, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_slli_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpslliq(a.as_i64x8(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpslliq(a.as_i64x8(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsrliq(a.as_i64x8(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srli_epi64&expand=5529)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlq, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srli_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsrliq(a.as_i64x8(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsrliq(a.as_i64x8(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsraid(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srai_epi32&expand=5434)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srai_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsraid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsraid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsraiq(a.as_i64x8(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 64-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srai_epi64&expand=5443)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsraq, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srai_epi64(src: __m512i, k: __mmask8, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsraiq(a.as_i64x8(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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
    macro_rules! call {
        ($imm8:expr) => {
            vpsraiq(a.as_i64x8(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
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

/// interval [1, 2)
pub const _MM_MANT_NORM_1_2: _MM_MANTISSA_NORM_ENUM = 0x00;
/// interval [0.5, 2)
pub const _MM_MANT_NORM_P5_2: _MM_MANTISSA_NORM_ENUM = 0x01;
/// interval [0.5, 1)
pub const _MM_MANT_NORM_P5_1: _MM_MANTISSA_NORM_ENUM = 0x02;
/// interval [0.75, 1.5)
pub const _MM_MANT_NORM_P75_1P5: _MM_MANTISSA_NORM_ENUM = 0x03;

/// sign = sign(SRC)
pub const _MM_MANT_SIGN_SRC: _MM_MANTISSA_SIGN_ENUM = 0x00;
/// sign = 0
pub const _MM_MANT_SIGN_ZERO: _MM_MANTISSA_SIGN_ENUM = 0x01;
/// DEST = NaN if sign(SRC) = 1
pub const _MM_MANT_SIGN_NAN: _MM_MANTISSA_SIGN_ENUM = 0x02;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.pmul.dq.512"]
    fn vpmuldq(a: i32x16, b: i32x16) -> i64x8;
    #[link_name = "llvm.x86.avx512.pmulu.dq.512"]
    fn vpmuludq(a: u32x16, b: u32x16) -> u64x8;

    #[link_name = "llvm.x86.avx512.mask.pmaxs.d.512"]
    fn vpmaxsd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.pmaxs.q.512"]
    fn vpmaxsq(a: i64x8, b: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pmins.d.512"]
    fn vpminsd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.pmins.q.512"]
    fn vpminsq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.pmaxu.d.512"]
    fn vpmaxud(a: u32x16, b: u32x16) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.pmaxu.q.512"]
    fn vpmaxuq(a: u64x8, b: u64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pminu.d.512"]
    fn vpminud(a: u32x16, b: u32x16) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.pminu.q.512"]
    fn vpminuq(a: u64x8, b: u64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.sqrt.ps.512"]
    fn vsqrtps(a: f32x16, rounding: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.sqrt.pd.512"]
    fn vsqrtpd(a: f64x8, rounding: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.vfmadd.ps.512"]
    fn vfmadd132ps(a: f32x16, b: f32x16, c: f32x16, rounding: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.vfmadd.pd.512"]
    fn vfmadd132pd(a: f64x8, b: f64x8, c: f64x8, rounding: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.vfmaddsub.ps.512"]
    fn vfmaddsub213ps(a: f32x16, b: f32x16, c: f32x16, d: i32) -> f32x16; //from clang
    #[link_name = "llvm.x86.avx512.vfmaddsub.pd.512"]
    fn vfmaddsub213pd(a: f64x8, b: f64x8, c: f64x8, d: i32) -> f64x8; //from clang

    #[link_name = "llvm.x86.avx512.add.ps.512"]
    fn vaddps(a: f32x16, b: f32x16, rounding: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.add.pd.512"]
    fn vaddpd(a: f64x8, b: f64x8, rounding: i32) -> f64x8;
    #[link_name = "llvm.x86.avx512.sub.ps.512"]
    fn vsubps(a: f32x16, b: f32x16, rounding: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.sub.pd.512"]
    fn vsubpd(a: f64x8, b: f64x8, rounding: i32) -> f64x8;
    #[link_name = "llvm.x86.avx512.mul.ps.512"]
    fn vmulps(a: f32x16, b: f32x16, rounding: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.mul.pd.512"]
    fn vmulpd(a: f64x8, b: f64x8, rounding: i32) -> f64x8;
    #[link_name = "llvm.x86.avx512.div.ps.512"]
    fn vdivps(a: f32x16, b: f32x16, rounding: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.div.pd.512"]
    fn vdivpd(a: f64x8, b: f64x8, rounding: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.max.ps.512"]
    fn vmaxps(a: f32x16, b: f32x16, sae: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.max.pd.512"]
    fn vmaxpd(a: f64x8, b: f64x8, sae: i32) -> f64x8;
    #[link_name = "llvm.x86.avx512.min.ps.512"]
    fn vminps(a: f32x16, b: f32x16, sae: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.min.pd.512"]
    fn vminpd(a: f64x8, b: f64x8, sae: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.getexp.ps.512"]
    fn vgetexpps(a: f32x16, src: f32x16, m: u16, sae: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.mask.getexp.pd.512"]
    fn vgetexppd(a: f64x8, src: f64x8, m: u8, sae: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.getmant.ps.512"]
    fn vgetmantps(a: f32x16, mantissas: i32, src: f32x16, m: u16, sae: i32) -> f32x16;
    #[link_name = "llvm.x86.avx512.mask.getmant.pd.512"]
    fn vgetmantpd(a: f64x8, mantissas: i32, src: f64x8, m: u8, sae: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.rcp14.ps.512"]
    fn vrcp14ps(a: f32x16, src: f32x16, m: u16) -> f32x16;
    #[link_name = "llvm.x86.avx512.rcp14.pd.512"]
    fn vrcp14pd(a: f64x8, src: f64x8, m: u8) -> f64x8;
    #[link_name = "llvm.x86.avx512.rsqrt14.ps.512"]
    fn vrsqrt14ps(a: f32x16, src: f32x16, m: u16) -> f32x16;
    #[link_name = "llvm.x86.avx512.rsqrt14.pd.512"]
    fn vrsqrt14pd(a: f64x8, src: f64x8, m: u8) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtps2dq.512"]
    fn vcvtps2dq(a: f32x16, src: i32x16, mask: u16, rounding: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.cvtps2udq.512"]
    fn vcvtps2udq(a: f32x16, src: u32x16, mask: u16, rounding: i32) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.cvtps2pd.512"]
    fn vcvtps2pd(a: f32x8, src: f64x8, mask: u8, sae: i32) -> f64x8;

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
        let r = _mm512_mask_abs_epi32(a, 0b00000000_11111111, a);
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
        let r = _mm512_maskz_abs_epi32(0b00000000_11111111, a);
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
    unsafe fn test_mm512_abs_ps() {
        #[rustfmt::skip]
        let a = _mm512_setr_ps(
            0., 1., -1., f32::MAX,
            f32::MIN, 100., -100., -32.,
            0., 1., -1., f32::MAX,
            f32::MIN, 100., -100., -32.,
        );
        let r = _mm512_abs_ps(a);
        let e = _mm512_setr_ps(
            0.,
            1.,
            1.,
            f32::MAX,
            f32::MAX,
            100.,
            100.,
            32.,
            0.,
            1.,
            1.,
            f32::MAX,
            f32::MAX,
            100.,
            100.,
            32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_abs_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let r = _mm512_mask_abs_ps(a, 0, a);
        assert_eq_m512(r, a);
        let r = _mm512_mask_abs_ps(a, 0b00000000_11111111, a);
        let e = _mm512_setr_ps(
            0.,
            1.,
            1.,
            f32::MAX,
            f32::MAX,
            100.,
            100.,
            32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_add_epi32(a, b);
        let e = _mm512_setr_epi32(
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_add_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_mask_add_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_add_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
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
    unsafe fn test_mm512_maskz_add_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_maskz_add_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_add_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
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
    unsafe fn test_mm512_add_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_add_ps(a, b);
        let e = _mm512_setr_ps(
            1.,
            2.,
            0.,
            f32::MAX,
            f32::MIN + 1.,
            101.,
            -99.,
            -31.,
            1.,
            2.,
            0.,
            f32::MAX,
            f32::MIN + 1.,
            101.,
            -99.,
            -31.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_add_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_mask_add_ps(a, 0, a, b);
        assert_eq_m512(r, a);
        let r = _mm512_mask_add_ps(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            1.,
            2.,
            0.,
            f32::MAX,
            f32::MIN + 1.,
            101.,
            -99.,
            -31.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_add_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_maskz_add_ps(0, a, b);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_add_ps(0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            1.,
            2.,
            0.,
            f32::MAX,
            f32::MIN + 1.,
            101.,
            -99.,
            -31.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sub_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_sub_epi32(a, b);
        let e = _mm512_setr_epi32(
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sub_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_mask_sub_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sub_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
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
    unsafe fn test_mm512_maskz_sub_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_maskz_sub_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sub_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
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
    unsafe fn test_mm512_sub_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_sub_ps(a, b);
        let e = _mm512_setr_ps(
            -1.,
            0.,
            -2.,
            f32::MAX - 1.,
            f32::MIN,
            99.,
            -101.,
            -33.,
            -1.,
            0.,
            -2.,
            f32::MAX - 1.,
            f32::MIN,
            99.,
            -101.,
            -33.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sub_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_mask_sub_ps(a, 0, a, b);
        assert_eq_m512(r, a);
        let r = _mm512_mask_sub_ps(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            -1.,
            0.,
            -2.,
            f32::MAX - 1.,
            f32::MIN,
            99.,
            -101.,
            -33.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sub_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_maskz_sub_ps(0, a, b);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_sub_ps(0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            -1.,
            0.,
            -2.,
            f32::MAX - 1.,
            f32::MIN,
            99.,
            -101.,
            -33.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mullo_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(2);
        let r = _mm512_mullo_epi32(a, b);
        let e = _mm512_setr_epi32(
            0, 2, -2, -2, 0, 200, -200, -64, 0, 2, -2, -2, 0, 200, -200, -64,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_mullo_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(2);
        let r = _mm512_mask_mullo_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mullo_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            0,
            2,
            -2,
            -2,
            0,
            200,
            -200,
            -64,
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
    unsafe fn test_mm512_maskz_mullo_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(2);
        let r = _mm512_maskz_mullo_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mullo_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 2, -2, -2, 0, 200, -200, -64, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mul_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(2.);
        let r = _mm512_mul_ps(a, b);
        let e = _mm512_setr_ps(
            0.,
            2.,
            -2.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            200.,
            -200.,
            -64.,
            0.,
            2.,
            -2.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            200.,
            -200.,
            -64.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_mul_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(2.);
        let r = _mm512_mask_mul_ps(a, 0, a, b);
        assert_eq_m512(r, a);
        let r = _mm512_mask_mul_ps(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            0.,
            2.,
            -2.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            200.,
            -200.,
            -64.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_mul_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
            0.,
            1.,
            -1.,
            f32::MAX,
            f32::MIN,
            100.,
            -100.,
            -32.,
        );
        let b = _mm512_set1_ps(2.);
        let r = _mm512_maskz_mul_ps(0, a, b);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_mul_ps(0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            0.,
            2.,
            -2.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            200.,
            -200.,
            -64.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_div_ps() {
        let a = _mm512_setr_ps(
            0., 1., -1., -2., 100., 100., -100., -32., 0., 1., -1., 1000., -131., 100., -100., -32.,
        );
        let b = _mm512_setr_ps(
            2., 2., 2., 2., 2., 0., 2., 2., 2., 2., 2., 2., 0., 2., 2., 2.,
        );
        let r = _mm512_div_ps(a, b);
        let e = _mm512_setr_ps(
            0.,
            0.5,
            -0.5,
            -1.,
            50.,
            f32::INFINITY,
            -50.,
            -16.,
            0.,
            0.5,
            -0.5,
            500.,
            f32::NEG_INFINITY,
            50.,
            -50.,
            -16.,
        );
        assert_eq_m512(r, e); // 0/0 = NAN
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_div_ps() {
        let a = _mm512_setr_ps(
            0., 1., -1., -2., 100., 100., -100., -32., 0., 1., -1., 1000., -131., 100., -100., -32.,
        );
        let b = _mm512_setr_ps(
            2., 2., 2., 2., 2., 0., 2., 2., 2., 2., 2., 2., 0., 2., 2., 2.,
        );
        let r = _mm512_mask_div_ps(a, 0, a, b);
        assert_eq_m512(r, a);
        let r = _mm512_mask_div_ps(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            0.,
            0.5,
            -0.5,
            -1.,
            50.,
            f32::INFINITY,
            -50.,
            -16.,
            0.,
            1.,
            -1.,
            1000.,
            -131.,
            100.,
            -100.,
            -32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_div_ps() {
        let a = _mm512_setr_ps(
            0., 1., -1., -2., 100., 100., -100., -32., 0., 1., -1., 1000., -131., 100., -100., -32.,
        );
        let b = _mm512_setr_ps(
            2., 2., 2., 2., 2., 0., 2., 2., 2., 2., 2., 2., 0., 2., 2., 2.,
        );
        let r = _mm512_maskz_div_ps(0, a, b);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_div_ps(0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            0.,
            0.5,
            -0.5,
            -1.,
            50.,
            f32::INFINITY,
            -50.,
            -16.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_max_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epi32(a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_max_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_max_ps(a, b);
        let e = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_mask_max_ps(a, 0, a, b);
        assert_eq_m512(r, a);
        let r = _mm512_mask_max_ps(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_maskz_max_ps(0, a, b);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_max_ps(0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_max_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epu32(a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epu32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epu32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epu32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epu32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_min_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epi32(a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_min_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_min_ps(a, b);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_mask_min_ps(a, 0, a, b);
        assert_eq_m512(r, a);
        let r = _mm512_mask_min_ps(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_maskz_min_ps(0, a, b);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_min_ps(0b00000000_11111111, a, b);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_min_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epu32(a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epu32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epu32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epu32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epu32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sqrt_ps() {
        let a = _mm512_setr_ps(
            0., 1., 4., 9., 16., 25., 36., 49., 64., 81., 100., 121., 144., 169., 196., 225.,
        );
        let r = _mm512_sqrt_ps(a);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sqrt_ps() {
        let a = _mm512_setr_ps(
            0., 1., 4., 9., 16., 25., 36., 49., 64., 81., 100., 121., 144., 169., 196., 225.,
        );
        let r = _mm512_mask_sqrt_ps(a, 0, a);
        assert_eq_m512(r, a);
        let r = _mm512_mask_sqrt_ps(a, 0b00000000_11111111, a);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 64., 81., 100., 121., 144., 169., 196., 225.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sqrt_ps() {
        let a = _mm512_setr_ps(
            0., 1., 4., 9., 16., 25., 36., 49., 64., 81., 100., 121., 144., 169., 196., 225.,
        );
        let r = _mm512_maskz_sqrt_ps(0, a);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_sqrt_ps(0b00000000_11111111, a);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_fmadd_ps(a, b, c);
        let e = _mm512_setr_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_mask_fmadd_ps(a, 0, b, c);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmadd_ps(a, 0b00000000_11111111, b, c);
        let e = _mm512_setr_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_maskz_fmadd_ps(0, a, b, c);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmadd_ps(0b00000000_11111111, a, b, c);
        let e = _mm512_setr_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        let r = _mm512_mask3_fmadd_ps(a, b, c, 0);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmadd_ps(a, b, c, 0b00000000_11111111);
        let e = _mm512_setr_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_fmsub_ps(a, b, c);
        let e = _mm512_setr_ps(
            -1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_mask_fmsub_ps(a, 0, b, c);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmsub_ps(a, 0b00000000_11111111, b, c);
        let e = _mm512_setr_ps(
            -1., 0., 1., 2., 3., 4., 5., 6., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_maskz_fmsub_ps(0, a, b, c);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmsub_ps(0b00000000_11111111, a, b, c);
        let e = _mm512_setr_ps(
            -1., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        let r = _mm512_mask3_fmsub_ps(a, b, c, 0);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmsub_ps(a, b, c, 0b00000000_11111111);
        let e = _mm512_setr_ps(
            -1., 0., 1., 2., 3., 4., 5., 6., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmaddsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_fmaddsub_ps(a, b, c);
        let e = _mm512_setr_ps(
            -1., 2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmaddsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_mask_fmaddsub_ps(a, 0, b, c);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmaddsub_ps(a, 0b00000000_11111111, b, c);
        let e = _mm512_setr_ps(
            -1., 2., 1., 4., 3., 6., 5., 8., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmaddsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_maskz_fmaddsub_ps(0, a, b, c);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmaddsub_ps(0b00000000_11111111, a, b, c);
        let e = _mm512_setr_ps(
            -1., 2., 1., 4., 3., 6., 5., 8., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmaddsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        let r = _mm512_mask3_fmaddsub_ps(a, b, c, 0);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmaddsub_ps(a, b, c, 0b00000000_11111111);
        let e = _mm512_setr_ps(
            -1., 2., 1., 4., 3., 6., 5., 8., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmsubadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_fmsubadd_ps(a, b, c);
        let e = _mm512_setr_ps(
            1., 0., 3., 2., 5., 4., 7., 6., 9., 8., 11., 10., 13., 12., 15., 14.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmsubadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_mask_fmsubadd_ps(a, 0, b, c);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmsubadd_ps(a, 0b00000000_11111111, b, c);
        let e = _mm512_setr_ps(
            1., 0., 3., 2., 5., 4., 7., 6., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmsubadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_maskz_fmsubadd_ps(0, a, b, c);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmsubadd_ps(0b00000000_11111111, a, b, c);
        let e = _mm512_setr_ps(
            1., 0., 3., 2., 5., 4., 7., 6., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmsubadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        let r = _mm512_mask3_fmsubadd_ps(a, b, c, 0);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmsubadd_ps(a, b, c, 0b00000000_11111111);
        let e = _mm512_setr_ps(
            1., 0., 3., 2., 5., 4., 7., 6., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fnmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_fnmadd_ps(a, b, c);
        let e = _mm512_setr_ps(
            1., 0., -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., -13., -14.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fnmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_mask_fnmadd_ps(a, 0, b, c);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fnmadd_ps(a, 0b00000000_11111111, b, c);
        let e = _mm512_setr_ps(
            1., 0., -1., -2., -3., -4., -5., -6., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fnmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_maskz_fnmadd_ps(0, a, b, c);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fnmadd_ps(0b00000000_11111111, a, b, c);
        let e = _mm512_setr_ps(
            1., 0., -1., -2., -3., -4., -5., -6., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fnmadd_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        let r = _mm512_mask3_fnmadd_ps(a, b, c, 0);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fnmadd_ps(a, b, c, 0b00000000_11111111);
        let e = _mm512_setr_ps(
            1., 0., -1., -2., -3., -4., -5., -6., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fnmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_fnmsub_ps(a, b, c);
        let e = _mm512_setr_ps(
            -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., -13., -14., -15., -16.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fnmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_mask_fnmsub_ps(a, 0, b, c);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fnmsub_ps(a, 0b00000000_11111111, b, c);
        let e = _mm512_setr_ps(
            -1., -2., -3., -4., -5., -6., -7., -8., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fnmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let r = _mm512_maskz_fnmsub_ps(0, a, b, c);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fnmsub_ps(0b00000000_11111111, a, b, c);
        let e = _mm512_setr_ps(
            -1., -2., -3., -4., -5., -6., -7., -8., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fnmsub_ps() {
        let a = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        let b = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let c = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        let r = _mm512_mask3_fnmsub_ps(a, b, c, 0);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fnmsub_ps(a, b, c, 0b00000000_11111111);
        let e = _mm512_setr_ps(
            -1., -2., -3., -4., -5., -6., -7., -8., 2., 2., 2., 2., 2., 2., 2., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_rcp14_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_rcp14_ps(a);
        let e = _mm512_set1_ps(0.33333206);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_rcp14_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_mask_rcp14_ps(a, 0, a);
        assert_eq_m512(r, a);
        let r = _mm512_mask_rcp14_ps(a, 0b11111111_00000000, a);
        let e = _mm512_setr_ps(
            3., 3., 3., 3., 3., 3., 3., 3., 0.33333206, 0.33333206, 0.33333206, 0.33333206,
            0.33333206, 0.33333206, 0.33333206, 0.33333206,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_rcp14_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_maskz_rcp14_ps(0, a);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_rcp14_ps(0b11111111_00000000, a);
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 0.33333206, 0.33333206, 0.33333206, 0.33333206,
            0.33333206, 0.33333206, 0.33333206, 0.33333206,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_rsqrt14_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_rsqrt14_ps(a);
        let e = _mm512_set1_ps(0.5773392);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_rsqrt14_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_mask_rsqrt14_ps(a, 0, a);
        assert_eq_m512(r, a);
        let r = _mm512_mask_rsqrt14_ps(a, 0b11111111_00000000, a);
        let e = _mm512_setr_ps(
            3., 3., 3., 3., 3., 3., 3., 3., 0.5773392, 0.5773392, 0.5773392, 0.5773392, 0.5773392,
            0.5773392, 0.5773392, 0.5773392,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_rsqrt14_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_maskz_rsqrt14_ps(0, a);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_rsqrt14_ps(0b11111111_00000000, a);
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 0.5773392, 0.5773392, 0.5773392, 0.5773392, 0.5773392,
            0.5773392, 0.5773392, 0.5773392,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_getexp_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_getexp_ps(a);
        let e = _mm512_set1_ps(1.);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_getexp_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_mask_getexp_ps(a, 0, a);
        assert_eq_m512(r, a);
        let r = _mm512_mask_getexp_ps(a, 0b11111111_00000000, a);
        let e = _mm512_setr_ps(
            3., 3., 3., 3., 3., 3., 3., 3., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_getexp_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_maskz_getexp_ps(0, a);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_getexp_ps(0b11111111_00000000, a);
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_getmant_ps() {
        let a = _mm512_set1_ps(10.);
        let r = _mm512_getmant_ps(a, _MM_MANT_NORM_P75_1P5, _MM_MANT_SIGN_NAN);
        let e = _mm512_set1_ps(1.25);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_getmant_ps() {
        let a = _mm512_set1_ps(10.);
        let r = _mm512_mask_getmant_ps(a, 0, a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_getmant_ps(
            a,
            0b11111111_00000000,
            a,
            _MM_MANT_NORM_1_2,
            _MM_MANT_SIGN_SRC,
        );
        let e = _mm512_setr_ps(
            10., 10., 10., 10., 10., 10., 10., 10., 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_getmant_ps() {
        let a = _mm512_set1_ps(10.);
        let r = _mm512_maskz_getmant_ps(0, a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r =
            _mm512_maskz_getmant_ps(0b11111111_00000000, a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC);
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_round_ps() {
        let a = _mm512_setr_ps(
            0., 1.5, 2., 3.5, 4., 5.5, 6., 7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 0.00000007,
        );
        let b = _mm512_set1_ps(-1.);
        let r = _mm512_add_round_ps(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            -1.,
            0.5,
            1.,
            2.5,
            3.,
            4.5,
            5.,
            6.5,
            7.,
            8.5,
            9.,
            10.5,
            11.,
            12.5,
            13.,
            -0.99999994,
        );
        assert_eq_m512(r, e);
        let r = _mm512_add_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            -1., 0.5, 1., 2.5, 3., 4.5, 5., 6.5, 7., 8.5, 9., 10.5, 11., 12.5, 13., -0.9999999,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_add_round_ps() {
        let a = _mm512_setr_ps(
            0., 1.5, 2., 3.5, 4., 5.5, 6., 7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 0.00000007,
        );
        let b = _mm512_set1_ps(-1.);
        let r = _mm512_mask_add_round_ps(a, 0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_add_round_ps(
            a,
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.,
            1.5,
            2.,
            3.5,
            4.,
            5.5,
            6.,
            7.5,
            7.,
            8.5,
            9.,
            10.5,
            11.,
            12.5,
            13.,
            -0.99999994,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_add_round_ps() {
        let a = _mm512_setr_ps(
            0., 1.5, 2., 3.5, 4., 5.5, 6., 7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 0.00000007,
        );
        let b = _mm512_set1_ps(-1.);
        let r = _mm512_maskz_add_round_ps(0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_add_round_ps(
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            7.,
            8.5,
            9.,
            10.5,
            11.,
            12.5,
            13.,
            -0.99999994,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sub_round_ps() {
        let a = _mm512_setr_ps(
            0., 1.5, 2., 3.5, 4., 5.5, 6., 7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 0.00000007,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_sub_round_ps(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            -1.,
            0.5,
            1.,
            2.5,
            3.,
            4.5,
            5.,
            6.5,
            7.,
            8.5,
            9.,
            10.5,
            11.,
            12.5,
            13.,
            -0.99999994,
        );
        assert_eq_m512(r, e);
        let r = _mm512_sub_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            -1., 0.5, 1., 2.5, 3., 4.5, 5., 6.5, 7., 8.5, 9., 10.5, 11., 12.5, 13., -0.9999999,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sub_round_ps() {
        let a = _mm512_setr_ps(
            0., 1.5, 2., 3.5, 4., 5.5, 6., 7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 0.00000007,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_mask_sub_round_ps(a, 0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_sub_round_ps(
            a,
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.,
            1.5,
            2.,
            3.5,
            4.,
            5.5,
            6.,
            7.5,
            7.,
            8.5,
            9.,
            10.5,
            11.,
            12.5,
            13.,
            -0.99999994,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sub_round_ps() {
        let a = _mm512_setr_ps(
            0., 1.5, 2., 3.5, 4., 5.5, 6., 7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 0.00000007,
        );
        let b = _mm512_set1_ps(1.);
        let r = _mm512_maskz_sub_round_ps(0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_sub_round_ps(
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            7.,
            8.5,
            9.,
            10.5,
            11.,
            12.5,
            13.,
            -0.99999994,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mul_round_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.5,
            2.,
            3.5,
            4.,
            5.5,
            6.,
            7.5,
            8.,
            9.5,
            10.,
            11.5,
            12.,
            13.5,
            14.,
            0.00000000000000000000007,
        );
        let b = _mm512_set1_ps(0.1);
        let r = _mm512_mul_round_ps(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            0.,
            0.15,
            0.2,
            0.35,
            0.4,
            0.55,
            0.6,
            0.75,
            0.8,
            0.95,
            1.0,
            1.15,
            1.2,
            1.35,
            1.4,
            0.000000000000000000000007000001,
        );
        assert_eq_m512(r, e);
        let r = _mm512_mul_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            0.,
            0.14999999,
            0.2,
            0.35,
            0.4,
            0.54999995,
            0.59999996,
            0.75,
            0.8,
            0.95,
            1.0,
            1.15,
            1.1999999,
            1.3499999,
            1.4,
            0.000000000000000000000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_mul_round_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.5,
            2.,
            3.5,
            4.,
            5.5,
            6.,
            7.5,
            8.,
            9.5,
            10.,
            11.5,
            12.,
            13.5,
            14.,
            0.00000000000000000000007,
        );
        let b = _mm512_set1_ps(0.1);
        let r = _mm512_mask_mul_round_ps(a, 0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_mul_round_ps(
            a,
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.,
            1.5,
            2.,
            3.5,
            4.,
            5.5,
            6.,
            7.5,
            0.8,
            0.95,
            1.0,
            1.15,
            1.2,
            1.35,
            1.4,
            0.000000000000000000000007000001,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_mul_round_ps() {
        let a = _mm512_setr_ps(
            0.,
            1.5,
            2.,
            3.5,
            4.,
            5.5,
            6.,
            7.5,
            8.,
            9.5,
            10.,
            11.5,
            12.,
            13.5,
            14.,
            0.00000000000000000000007,
        );
        let b = _mm512_set1_ps(0.1);
        let r = _mm512_maskz_mul_round_ps(0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_mul_round_ps(
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.8,
            0.95,
            1.0,
            1.15,
            1.2,
            1.35,
            1.4,
            0.000000000000000000000007000001,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_div_round_ps() {
        let a = _mm512_set1_ps(1.);
        let b = _mm512_set1_ps(3.);
        let r = _mm512_div_round_ps(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(0.33333334);
        assert_eq_m512(r, e);
        let r = _mm512_div_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(0.3333333);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_div_round_ps() {
        let a = _mm512_set1_ps(1.);
        let b = _mm512_set1_ps(3.);
        let r = _mm512_mask_div_round_ps(a, 0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_div_round_ps(
            a,
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            1., 1., 1., 1., 1., 1., 1., 1., 0.33333334, 0.33333334, 0.33333334, 0.33333334,
            0.33333334, 0.33333334, 0.33333334, 0.33333334,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_div_round_ps() {
        let a = _mm512_set1_ps(1.);
        let b = _mm512_set1_ps(3.);
        let r = _mm512_maskz_div_round_ps(0, a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_div_round_ps(
            0b11111111_00000000,
            a,
            b,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 0.33333334, 0.33333334, 0.33333334, 0.33333334,
            0.33333334, 0.33333334, 0.33333334, 0.33333334,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sqrt_round_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_sqrt_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(1.7320508);
        assert_eq_m512(r, e);
        let r = _mm512_sqrt_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(1.7320509);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sqrt_round_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_mask_sqrt_round_ps(a, 0, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_sqrt_round_ps(
            a,
            0b11111111_00000000,
            a,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            3., 3., 3., 3., 3., 3., 3., 3., 1.7320508, 1.7320508, 1.7320508, 1.7320508, 1.7320508,
            1.7320508, 1.7320508, 1.7320508,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sqrt_round_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_maskz_sqrt_round_ps(0, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_sqrt_round_ps(
            0b11111111_00000000,
            a,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 1.7320508, 1.7320508, 1.7320508, 1.7320508, 1.7320508,
            1.7320508, 1.7320508, 1.7320508,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(-0.99999994);
        assert_eq_m512(r, e);
        let r = _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(-0.9999999);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r =
            _mm512_mask_fmadd_round_ps(a, 0, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmadd_round_ps(
            a,
            0b00000000_11111111,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r =
            _mm512_maskz_fmadd_round_ps(0, a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmadd_round_ps(
            0b00000000_11111111,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r =
            _mm512_mask3_fmadd_round_ps(a, b, c, 0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmadd_round_ps(
            a,
            b,
            c,
            0b00000000_11111111,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r = _mm512_fmsub_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(-0.99999994);
        assert_eq_m512(r, e);
        let r = _mm512_fmsub_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(-0.9999999);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r =
            _mm512_mask_fmsub_round_ps(a, 0, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmsub_round_ps(
            a,
            0b00000000_11111111,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r =
            _mm512_maskz_fmsub_round_ps(0, a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmsub_round_ps(
            0b00000000_11111111,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r =
            _mm512_mask3_fmsub_round_ps(a, b, c, 0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmsub_round_ps(
            a,
            b,
            c,
            0b00000000_11111111,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            -0.99999994,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmaddsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_fmaddsub_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
        );
        assert_eq_m512(r, e);
        let r = _mm512_fmaddsub_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            1., -0.9999999, 1., -0.9999999, 1., -0.9999999, 1., -0.9999999, 1., -0.9999999, 1.,
            -0.9999999, 1., -0.9999999, 1., -0.9999999,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmaddsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_mask_fmaddsub_round_ps(
            a,
            0,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmaddsub_round_ps(
            a,
            0b00000000_11111111,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmaddsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_maskz_fmaddsub_round_ps(
            0,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmaddsub_round_ps(
            0b00000000_11111111,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmaddsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_mask3_fmaddsub_round_ps(
            a,
            b,
            c,
            0,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmaddsub_round_ps(
            a,
            b,
            c,
            0b00000000_11111111,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fmsubadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_fmsubadd_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
        );
        assert_eq_m512(r, e);
        let r = _mm512_fmsubadd_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_ps(
            -0.9999999, 1., -0.9999999, 1., -0.9999999, 1., -0.9999999, 1., -0.9999999, 1.,
            -0.9999999, 1., -0.9999999, 1., -0.9999999, 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fmsubadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_mask_fmsubadd_round_ps(
            a,
            0,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        assert_eq_m512(r, a);
        let r = _mm512_mask_fmsubadd_round_ps(
            a,
            0b00000000_11111111,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
            0.00000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fmsubadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_maskz_fmsubadd_round_ps(
            0,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fmsubadd_round_ps(
            0b00000000_11111111,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fmsubadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_mask3_fmsubadd_round_ps(
            a,
            b,
            c,
            0,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fmsubadd_round_ps(
            a,
            b,
            c,
            0b00000000_11111111,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -0.99999994,
            1.0000001,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fnmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r = _mm512_fnmadd_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(0.99999994);
        assert_eq_m512(r, e);
        let r = _mm512_fnmadd_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(0.9999999);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fnmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r =
            _mm512_mask_fnmadd_round_ps(a, 0, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fnmadd_round_ps(
            a,
            0b00000000_11111111,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994,
            0.99999994, 0.00000007, 0.00000007, 0.00000007, 0.00000007, 0.00000007, 0.00000007,
            0.00000007, 0.00000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fnmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r =
            _mm512_maskz_fnmadd_round_ps(0, a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fnmadd_round_ps(
            0b00000000_11111111,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994,
            0.99999994, 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fnmadd_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(1.);
        let r =
            _mm512_mask3_fnmadd_round_ps(a, b, c, 0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fnmadd_round_ps(
            a,
            b,
            c,
            0b00000000_11111111,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994,
            0.99999994, 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_fnmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r = _mm512_fnmsub_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(0.99999994);
        assert_eq_m512(r, e);
        let r = _mm512_fnmsub_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(0.9999999);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_fnmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r =
            _mm512_mask_fnmsub_round_ps(a, 0, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, a);
        let r = _mm512_mask_fnmsub_round_ps(
            a,
            0b00000000_11111111,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994,
            0.99999994, 0.00000007, 0.00000007, 0.00000007, 0.00000007, 0.00000007, 0.00000007,
            0.00000007, 0.00000007,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_fnmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r =
            _mm512_maskz_fnmsub_round_ps(0, a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_fnmsub_round_ps(
            0b00000000_11111111,
            a,
            b,
            c,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994,
            0.99999994, 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask3_fnmsub_round_ps() {
        let a = _mm512_set1_ps(0.00000007);
        let b = _mm512_set1_ps(1.);
        let c = _mm512_set1_ps(-1.);
        let r =
            _mm512_mask3_fnmsub_round_ps(a, b, c, 0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512(r, c);
        let r = _mm512_mask3_fnmsub_round_ps(
            a,
            b,
            c,
            0b00000000_11111111,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_ps(
            0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994, 0.99999994,
            0.99999994, -1., -1., -1., -1., -1., -1., -1., -1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_max_round_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_max_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_round_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_mask_max_round_ps(a, 0, a, b, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m512(r, a);
        let r = _mm512_mask_max_round_ps(a, 0b00000000_11111111, a, b, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_round_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_maskz_max_round_ps(0, a, b, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_max_round_ps(0b00000000_11111111, a, b, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_min_round_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_min_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_round_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_mask_min_round_ps(a, 0, a, b, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m512(r, a);
        let r = _mm512_mask_min_round_ps(a, 0b00000000_11111111, a, b, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_round_ps() {
        let a = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        );
        let b = _mm512_setr_ps(
            15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
        );
        let r = _mm512_maskz_min_round_ps(0, a, b, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_min_round_ps(0b00000000_11111111, a, b, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            0., 1., 2., 3., 4., 5., 6., 7., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_getexp_round_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_getexp_round_ps(a, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_set1_ps(1.);
        assert_eq_m512(r, e);
        let r = _mm512_getexp_round_ps(a, _MM_FROUND_NO_EXC);
        let e = _mm512_set1_ps(1.);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_getexp_round_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_mask_getexp_round_ps(a, 0, a, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m512(r, a);
        let r = _mm512_mask_getexp_round_ps(a, 0b11111111_00000000, a, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            3., 3., 3., 3., 3., 3., 3., 3., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_getexp_round_ps() {
        let a = _mm512_set1_ps(3.);
        let r = _mm512_maskz_getexp_round_ps(0, a, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_getexp_round_ps(0b11111111_00000000, a, _MM_FROUND_CUR_DIRECTION);
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_getmant_round_ps() {
        let a = _mm512_set1_ps(10.);
        let r = _mm512_getmant_round_ps(
            a,
            _MM_MANT_NORM_1_2,
            _MM_MANT_SIGN_SRC,
            _MM_FROUND_CUR_DIRECTION,
        );
        let e = _mm512_set1_ps(1.25);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_getmant_round_ps() {
        let a = _mm512_set1_ps(10.);
        let r = _mm512_mask_getmant_round_ps(
            a,
            0,
            a,
            _MM_MANT_NORM_1_2,
            _MM_MANT_SIGN_SRC,
            _MM_FROUND_CUR_DIRECTION,
        );
        assert_eq_m512(r, a);
        let r = _mm512_mask_getmant_round_ps(
            a,
            0b11111111_00000000,
            a,
            _MM_MANT_NORM_1_2,
            _MM_MANT_SIGN_SRC,
            _MM_FROUND_CUR_DIRECTION,
        );
        let e = _mm512_setr_ps(
            10., 10., 10., 10., 10., 10., 10., 10., 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_getmant_round_ps() {
        let a = _mm512_set1_ps(10.);
        let r = _mm512_maskz_getmant_round_ps(
            0,
            a,
            _MM_MANT_NORM_1_2,
            _MM_MANT_SIGN_SRC,
            _MM_FROUND_CUR_DIRECTION,
        );
        assert_eq_m512(r, _mm512_setzero_ps());
        let r = _mm512_maskz_getmant_round_ps(
            0b11111111_00000000,
            a,
            _MM_MANT_NORM_1_2,
            _MM_MANT_SIGN_SRC,
            _MM_FROUND_CUR_DIRECTION,
        );
        let e = _mm512_setr_ps(
            0., 0., 0., 0., 0., 0., 0., 0., 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtps_epi32() {
        let a = _mm512_setr_ps(
            0., -1.4, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_cvtps_epi32(a);
        let e = _mm512_setr_epi32(0, -1, 2, -4, 4, -6, 6, -8, 8, 10, 10, 12, 12, 14, 14, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtps_epi32() {
        let a = _mm512_setr_ps(
            0., -1.4, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let src = _mm512_set1_epi32(0);
        let r = _mm512_mask_cvtps_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtps_epi32(src, 0b00000000_11111111, a);
        let e = _mm512_setr_epi32(0, -1, 2, -4, 4, -6, 6, -8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtps_epi32() {
        let a = _mm512_setr_ps(
            0., -1.4, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_maskz_cvtps_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtps_epi32(0b00000000_11111111, a);
        let e = _mm512_setr_epi32(0, -1, 2, -4, 4, -6, 6, -8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtps_epu32() {
        let a = _mm512_setr_ps(
            0., -1.4, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_cvtps_epu32(a);
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 8, 10, 10, 12, 12, 14, 14, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtps_epu32() {
        let a = _mm512_setr_ps(
            0., -1.4, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let src = _mm512_set1_epi32(0);
        let r = _mm512_mask_cvtps_epu32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtps_epu32(src, 0b00000000_11111111, a);
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtps_epu32() {
        let a = _mm512_setr_ps(
            0., -1.4, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_maskz_cvtps_epu32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtps_epu32(0b00000000_11111111, a);
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvt_roundps_epi32() {
        let a = _mm512_setr_ps(
            0., -1.5, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_cvt_roundps_epi32(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_epi32(0, -2, 2, -4, 4, -6, 6, -8, 8, 10, 10, 12, 12, 14, 14, 16);
        assert_eq_m512i(r, e);
        let r = _mm512_cvt_roundps_epi32(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_epi32(0, -2, 2, -4, 4, -6, 6, -8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvt_roundps_epi32() {
        let a = _mm512_setr_ps(
            0., -1.5, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let src = _mm512_set1_epi32(0);
        let r =
            _mm512_mask_cvt_roundps_epi32(src, 0, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvt_roundps_epi32(
            src,
            0b00000000_11111111,
            a,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_epi32(0, -2, 2, -4, 4, -6, 6, -8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvt_roundps_epi32() {
        let a = _mm512_setr_ps(
            0., -1.5, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_maskz_cvt_roundps_epi32(0, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvt_roundps_epi32(
            0b00000000_11111111,
            a,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_epi32(0, -2, 2, -4, 4, -6, 6, -8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvt_roundps_epu32() {
        let a = _mm512_setr_ps(
            0., -1.5, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_cvt_roundps_epu32(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 8, 10, 10, 12, 12, 14, 14, 16);
        assert_eq_m512i(r, e);
        let r = _mm512_cvt_roundps_epu32(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvt_roundps_epu32() {
        let a = _mm512_setr_ps(
            0., -1.5, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let src = _mm512_set1_epi32(0);
        let r =
            _mm512_mask_cvt_roundps_epu32(src, 0, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvt_roundps_epu32(
            src,
            0b00000000_11111111,
            a,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvt_roundps_epu32() {
        let a = _mm512_setr_ps(
            0., -1.5, 2., -3.5, 4., -5.5, 6., -7.5, 8., 9.5, 10., 11.5, 12., 13.5, 14., 15.5,
        );
        let r = _mm512_maskz_cvt_roundps_epu32(0, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvt_roundps_epu32(
            0b00000000_11111111,
            a,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let e = _mm512_setr_epi32(0, -1, 2, -1, 4, -1, 6, -1, 0, 0, 0, 0, 0, 0, 0, 0);
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

use crate::core_arch::{simd::*, x86::*};
use crate::intrinsics::simd::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Broadcast the low 16-bits from input mask k to all 32-bit elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcastmw_epi32&expand=553)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmw2d
pub fn _mm512_broadcastmw_epi32(k: __mmask16) -> __m512i {
    _mm512_set1_epi32(k as i32)
}

/// Broadcast the low 16-bits from input mask k to all 32-bit elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcastmw_epi32&expand=552)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmw2d
pub fn _mm256_broadcastmw_epi32(k: __mmask16) -> __m256i {
    _mm256_set1_epi32(k as i32)
}

/// Broadcast the low 16-bits from input mask k to all 32-bit elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_broadcastmw_epi32&expand=551)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmw2d
pub fn _mm_broadcastmw_epi32(k: __mmask16) -> __m128i {
    _mm_set1_epi32(k as i32)
}

/// Broadcast the low 8-bits from input mask k to all 64-bit elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcastmb_epi64&expand=550)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmb2q
pub fn _mm512_broadcastmb_epi64(k: __mmask8) -> __m512i {
    _mm512_set1_epi64(k as i64)
}

/// Broadcast the low 8-bits from input mask k to all 64-bit elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcastmb_epi64&expand=549)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmb2q
pub fn _mm256_broadcastmb_epi64(k: __mmask8) -> __m256i {
    _mm256_set1_epi64x(k as i64)
}

/// Broadcast the low 8-bits from input mask k to all 64-bit elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_broadcastmb_epi64&expand=548)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmb2q
pub fn _mm_broadcastmb_epi64(k: __mmask8) -> __m128i {
    _mm_set1_epi64x(k as i64)
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_conflict_epi32&expand=1248)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm512_conflict_epi32(a: __m512i) -> __m512i {
    unsafe { transmute(vpconflictd(a.as_i32x16())) }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_conflict_epi32&expand=1249)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm512_mask_conflict_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    unsafe {
        let conflict = _mm512_conflict_epi32(a).as_i32x16();
        transmute(simd_select_bitmask(k, conflict, src.as_i32x16()))
    }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_conflict_epi32&expand=1250)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm512_maskz_conflict_epi32(k: __mmask16, a: __m512i) -> __m512i {
    unsafe {
        let conflict = _mm512_conflict_epi32(a).as_i32x16();
        transmute(simd_select_bitmask(k, conflict, i32x16::ZERO))
    }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_conflict_epi32&expand=1245)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm256_conflict_epi32(a: __m256i) -> __m256i {
    unsafe { transmute(vpconflictd256(a.as_i32x8())) }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_conflict_epi32&expand=1246)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm256_mask_conflict_epi32(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let conflict = _mm256_conflict_epi32(a).as_i32x8();
        transmute(simd_select_bitmask(k, conflict, src.as_i32x8()))
    }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_conflict_epi32&expand=1247)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm256_maskz_conflict_epi32(k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let conflict = _mm256_conflict_epi32(a).as_i32x8();
        transmute(simd_select_bitmask(k, conflict, i32x8::ZERO))
    }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_conflict_epi32&expand=1242)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm_conflict_epi32(a: __m128i) -> __m128i {
    unsafe { transmute(vpconflictd128(a.as_i32x4())) }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_conflict_epi32&expand=1243)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm_mask_conflict_epi32(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let conflict = _mm_conflict_epi32(a).as_i32x4();
        transmute(simd_select_bitmask(k, conflict, src.as_i32x4()))
    }
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_conflict_epi32&expand=1244)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub fn _mm_maskz_conflict_epi32(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let conflict = _mm_conflict_epi32(a).as_i32x4();
        transmute(simd_select_bitmask(k, conflict, i32x4::ZERO))
    }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_conflict_epi64&expand=1257)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm512_conflict_epi64(a: __m512i) -> __m512i {
    unsafe { transmute(vpconflictq(a.as_i64x8())) }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_conflict_epi64&expand=1258)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm512_mask_conflict_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    unsafe {
        let conflict = _mm512_conflict_epi64(a).as_i64x8();
        transmute(simd_select_bitmask(k, conflict, src.as_i64x8()))
    }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_conflict_epi64&expand=1259)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm512_maskz_conflict_epi64(k: __mmask8, a: __m512i) -> __m512i {
    unsafe {
        let conflict = _mm512_conflict_epi64(a).as_i64x8();
        transmute(simd_select_bitmask(k, conflict, i64x8::ZERO))
    }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_conflict_epi64&expand=1254)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm256_conflict_epi64(a: __m256i) -> __m256i {
    unsafe { transmute(vpconflictq256(a.as_i64x4())) }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_conflict_epi64&expand=1255)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm256_mask_conflict_epi64(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let conflict = _mm256_conflict_epi64(a).as_i64x4();
        transmute(simd_select_bitmask(k, conflict, src.as_i64x4()))
    }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_conflict_epi64&expand=1256)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm256_maskz_conflict_epi64(k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let conflict = _mm256_conflict_epi64(a).as_i64x4();
        transmute(simd_select_bitmask(k, conflict, i64x4::ZERO))
    }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_conflict_epi64&expand=1251)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm_conflict_epi64(a: __m128i) -> __m128i {
    unsafe { transmute(vpconflictq128(a.as_i64x2())) }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_conflict_epi64&expand=1252)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm_mask_conflict_epi64(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let conflict = _mm_conflict_epi64(a).as_i64x2();
        transmute(simd_select_bitmask(k, conflict, src.as_i64x2()))
    }
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_conflict_epi64&expand=1253)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub fn _mm_maskz_conflict_epi64(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let conflict = _mm_conflict_epi64(a).as_i64x2();
        transmute(simd_select_bitmask(k, conflict, i64x2::ZERO))
    }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_lzcnt_epi32&expand=3491)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm512_lzcnt_epi32(a: __m512i) -> __m512i {
    unsafe { transmute(simd_ctlz(a.as_i32x16())) }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_lzcnt_epi32&expand=3492)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm512_mask_lzcnt_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    unsafe {
        let zerocount = _mm512_lzcnt_epi32(a).as_i32x16();
        transmute(simd_select_bitmask(k, zerocount, src.as_i32x16()))
    }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_lzcnt_epi32&expand=3493)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm512_maskz_lzcnt_epi32(k: __mmask16, a: __m512i) -> __m512i {
    unsafe {
        let zerocount = _mm512_lzcnt_epi32(a).as_i32x16();
        transmute(simd_select_bitmask(k, zerocount, i32x16::ZERO))
    }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_lzcnt_epi32&expand=3488)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm256_lzcnt_epi32(a: __m256i) -> __m256i {
    unsafe { transmute(simd_ctlz(a.as_i32x8())) }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_lzcnt_epi32&expand=3489)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm256_mask_lzcnt_epi32(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let zerocount = _mm256_lzcnt_epi32(a).as_i32x8();
        transmute(simd_select_bitmask(k, zerocount, src.as_i32x8()))
    }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_lzcnt_epi32&expand=3490)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm256_maskz_lzcnt_epi32(k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let zerocount = _mm256_lzcnt_epi32(a).as_i32x8();
        transmute(simd_select_bitmask(k, zerocount, i32x8::ZERO))
    }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_lzcnt_epi32&expand=3485)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm_lzcnt_epi32(a: __m128i) -> __m128i {
    unsafe { transmute(simd_ctlz(a.as_i32x4())) }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_lzcnt_epi32&expand=3486)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm_mask_lzcnt_epi32(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let zerocount = _mm_lzcnt_epi32(a).as_i32x4();
        transmute(simd_select_bitmask(k, zerocount, src.as_i32x4()))
    }
}

/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_lzcnt_epi32&expand=3487)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub fn _mm_maskz_lzcnt_epi32(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let zerocount = _mm_lzcnt_epi32(a).as_i32x4();
        transmute(simd_select_bitmask(k, zerocount, i32x4::ZERO))
    }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_lzcnt_epi64&expand=3500)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm512_lzcnt_epi64(a: __m512i) -> __m512i {
    unsafe { transmute(simd_ctlz(a.as_i64x8())) }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_lzcnt_epi64&expand=3501)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm512_mask_lzcnt_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    unsafe {
        let zerocount = _mm512_lzcnt_epi64(a).as_i64x8();
        transmute(simd_select_bitmask(k, zerocount, src.as_i64x8()))
    }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_lzcnt_epi64&expand=3502)
#[inline]
#[target_feature(enable = "avx512cd")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm512_maskz_lzcnt_epi64(k: __mmask8, a: __m512i) -> __m512i {
    unsafe {
        let zerocount = _mm512_lzcnt_epi64(a).as_i64x8();
        transmute(simd_select_bitmask(k, zerocount, i64x8::ZERO))
    }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_lzcnt_epi64&expand=3497)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm256_lzcnt_epi64(a: __m256i) -> __m256i {
    unsafe { transmute(simd_ctlz(a.as_i64x4())) }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_lzcnt_epi64&expand=3498)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm256_mask_lzcnt_epi64(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let zerocount = _mm256_lzcnt_epi64(a).as_i64x4();
        transmute(simd_select_bitmask(k, zerocount, src.as_i64x4()))
    }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_lzcnt_epi64&expand=3499)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm256_maskz_lzcnt_epi64(k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        let zerocount = _mm256_lzcnt_epi64(a).as_i64x4();
        transmute(simd_select_bitmask(k, zerocount, i64x4::ZERO))
    }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_lzcnt_epi64&expand=3494)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm_lzcnt_epi64(a: __m128i) -> __m128i {
    unsafe { transmute(simd_ctlz(a.as_i64x2())) }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_lzcnt_epi64&expand=3495)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm_mask_lzcnt_epi64(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let zerocount = _mm_lzcnt_epi64(a).as_i64x2();
        transmute(simd_select_bitmask(k, zerocount, src.as_i64x2()))
    }
}

/// Counts the number of leading zero bits in each packed 64-bit integer in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_lzcnt_epi64&expand=3496)
#[inline]
#[target_feature(enable = "avx512cd,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vplzcntq))]
pub fn _mm_maskz_lzcnt_epi64(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let zerocount = _mm_lzcnt_epi64(a).as_i64x2();
        transmute(simd_select_bitmask(k, zerocount, i64x2::ZERO))
    }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512.conflict.d.512"]
    fn vpconflictd(a: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.conflict.d.256"]
    fn vpconflictd256(a: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx512.conflict.d.128"]
    fn vpconflictd128(a: i32x4) -> i32x4;

    #[link_name = "llvm.x86.avx512.conflict.q.512"]
    fn vpconflictq(a: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.conflict.q.256"]
    fn vpconflictq256(a: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx512.conflict.q.128"]
    fn vpconflictq128(a: i64x2) -> i64x2;
}

#[cfg(test)]
mod tests {

    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_broadcastmw_epi32() {
        let a: __mmask16 = 2;
        let r = _mm512_broadcastmw_epi32(a);
        let e = _mm512_set1_epi32(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_broadcastmw_epi32() {
        let a: __mmask16 = 2;
        let r = _mm256_broadcastmw_epi32(a);
        let e = _mm256_set1_epi32(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_broadcastmw_epi32() {
        let a: __mmask16 = 2;
        let r = _mm_broadcastmw_epi32(a);
        let e = _mm_set1_epi32(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_broadcastmb_epi64() {
        let a: __mmask8 = 2;
        let r = _mm512_broadcastmb_epi64(a);
        let e = _mm512_set1_epi64(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_broadcastmb_epi64() {
        let a: __mmask8 = 2;
        let r = _mm256_broadcastmb_epi64(a);
        let e = _mm256_set1_epi64x(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_broadcastmb_epi64() {
        let a: __mmask8 = 2;
        let r = _mm_broadcastmb_epi64(a);
        let e = _mm_set1_epi64x(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_conflict_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_conflict_epi32(a);
        let e = _mm512_set_epi32(
            1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_mask_conflict_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_mask_conflict_epi32(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_conflict_epi32(a, 0b11111111_11111111, a);
        let e = _mm512_set_epi32(
            1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_maskz_conflict_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_maskz_conflict_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_conflict_epi32(0b11111111_11111111, a);
        let e = _mm512_set_epi32(
            1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_conflict_epi32() {
        let a = _mm256_set1_epi32(1);
        let r = _mm256_conflict_epi32(a);
        let e = _mm256_set_epi32(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_mask_conflict_epi32() {
        let a = _mm256_set1_epi32(1);
        let r = _mm256_mask_conflict_epi32(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_conflict_epi32(a, 0b11111111, a);
        let e = _mm256_set_epi32(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_maskz_conflict_epi32() {
        let a = _mm256_set1_epi32(1);
        let r = _mm256_maskz_conflict_epi32(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_conflict_epi32(0b11111111, a);
        let e = _mm256_set_epi32(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_conflict_epi32() {
        let a = _mm_set1_epi32(1);
        let r = _mm_conflict_epi32(a);
        let e = _mm_set_epi32(1 << 2 | 1 << 1 | 1 << 0, 1 << 1 | 1 << 0, 1 << 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_mask_conflict_epi32() {
        let a = _mm_set1_epi32(1);
        let r = _mm_mask_conflict_epi32(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_conflict_epi32(a, 0b00001111, a);
        let e = _mm_set_epi32(1 << 2 | 1 << 1 | 1 << 0, 1 << 1 | 1 << 0, 1 << 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_maskz_conflict_epi32() {
        let a = _mm_set1_epi32(1);
        let r = _mm_maskz_conflict_epi32(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_conflict_epi32(0b00001111, a);
        let e = _mm_set_epi32(1 << 2 | 1 << 1 | 1 << 0, 1 << 1 | 1 << 0, 1 << 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_conflict_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_conflict_epi64(a);
        let e = _mm512_set_epi64(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_mask_conflict_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_mask_conflict_epi64(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_conflict_epi64(a, 0b11111111, a);
        let e = _mm512_set_epi64(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_maskz_conflict_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_maskz_conflict_epi64(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_conflict_epi64(0b11111111, a);
        let e = _mm512_set_epi64(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_conflict_epi64() {
        let a = _mm256_set1_epi64x(1);
        let r = _mm256_conflict_epi64(a);
        let e = _mm256_set_epi64x(1 << 2 | 1 << 1 | 1 << 0, 1 << 1 | 1 << 0, 1 << 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_mask_conflict_epi64() {
        let a = _mm256_set1_epi64x(1);
        let r = _mm256_mask_conflict_epi64(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_conflict_epi64(a, 0b00001111, a);
        let e = _mm256_set_epi64x(1 << 2 | 1 << 1 | 1 << 0, 1 << 1 | 1 << 0, 1 << 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_maskz_conflict_epi64() {
        let a = _mm256_set1_epi64x(1);
        let r = _mm256_maskz_conflict_epi64(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_conflict_epi64(0b00001111, a);
        let e = _mm256_set_epi64x(1 << 2 | 1 << 1 | 1 << 0, 1 << 1 | 1 << 0, 1 << 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_conflict_epi64() {
        let a = _mm_set1_epi64x(1);
        let r = _mm_conflict_epi64(a);
        let e = _mm_set_epi64x(1 << 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_mask_conflict_epi64() {
        let a = _mm_set1_epi64x(1);
        let r = _mm_mask_conflict_epi64(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_conflict_epi64(a, 0b00000011, a);
        let e = _mm_set_epi64x(1 << 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_maskz_conflict_epi64() {
        let a = _mm_set1_epi64x(1);
        let r = _mm_maskz_conflict_epi64(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_conflict_epi64(0b00000011, a);
        let e = _mm_set_epi64x(1 << 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_lzcnt_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_lzcnt_epi32(a);
        let e = _mm512_set1_epi32(31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_mask_lzcnt_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_mask_lzcnt_epi32(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_lzcnt_epi32(a, 0b11111111_11111111, a);
        let e = _mm512_set1_epi32(31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_maskz_lzcnt_epi32() {
        let a = _mm512_set1_epi32(2);
        let r = _mm512_maskz_lzcnt_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_lzcnt_epi32(0b11111111_11111111, a);
        let e = _mm512_set1_epi32(30);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_lzcnt_epi32() {
        let a = _mm256_set1_epi32(1);
        let r = _mm256_lzcnt_epi32(a);
        let e = _mm256_set1_epi32(31);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_mask_lzcnt_epi32() {
        let a = _mm256_set1_epi32(1);
        let r = _mm256_mask_lzcnt_epi32(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_lzcnt_epi32(a, 0b11111111, a);
        let e = _mm256_set1_epi32(31);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_maskz_lzcnt_epi32() {
        let a = _mm256_set1_epi32(1);
        let r = _mm256_maskz_lzcnt_epi32(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_lzcnt_epi32(0b11111111, a);
        let e = _mm256_set1_epi32(31);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_lzcnt_epi32() {
        let a = _mm_set1_epi32(1);
        let r = _mm_lzcnt_epi32(a);
        let e = _mm_set1_epi32(31);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_mask_lzcnt_epi32() {
        let a = _mm_set1_epi32(1);
        let r = _mm_mask_lzcnt_epi32(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_lzcnt_epi32(a, 0b00001111, a);
        let e = _mm_set1_epi32(31);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_maskz_lzcnt_epi32() {
        let a = _mm_set1_epi32(1);
        let r = _mm_maskz_lzcnt_epi32(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_lzcnt_epi32(0b00001111, a);
        let e = _mm_set1_epi32(31);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_lzcnt_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_lzcnt_epi64(a);
        let e = _mm512_set1_epi64(63);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_mask_lzcnt_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_mask_lzcnt_epi64(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_lzcnt_epi64(a, 0b11111111, a);
        let e = _mm512_set1_epi64(63);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_maskz_lzcnt_epi64() {
        let a = _mm512_set1_epi64(2);
        let r = _mm512_maskz_lzcnt_epi64(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_lzcnt_epi64(0b11111111, a);
        let e = _mm512_set1_epi64(62);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_lzcnt_epi64() {
        let a = _mm256_set1_epi64x(1);
        let r = _mm256_lzcnt_epi64(a);
        let e = _mm256_set1_epi64x(63);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_mask_lzcnt_epi64() {
        let a = _mm256_set1_epi64x(1);
        let r = _mm256_mask_lzcnt_epi64(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_lzcnt_epi64(a, 0b00001111, a);
        let e = _mm256_set1_epi64x(63);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm256_maskz_lzcnt_epi64() {
        let a = _mm256_set1_epi64x(1);
        let r = _mm256_maskz_lzcnt_epi64(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_lzcnt_epi64(0b00001111, a);
        let e = _mm256_set1_epi64x(63);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_lzcnt_epi64() {
        let a = _mm_set1_epi64x(1);
        let r = _mm_lzcnt_epi64(a);
        let e = _mm_set1_epi64x(63);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_mask_lzcnt_epi64() {
        let a = _mm_set1_epi64x(1);
        let r = _mm_mask_lzcnt_epi64(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_lzcnt_epi64(a, 0b00001111, a);
        let e = _mm_set1_epi64x(63);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512cd,avx512vl")]
    unsafe fn test_mm_maskz_lzcnt_epi64() {
        let a = _mm_set1_epi64x(1);
        let r = _mm_maskz_lzcnt_epi64(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_lzcnt_epi64(0b00001111, a);
        let e = _mm_set1_epi64x(63);
        assert_eq_m128i(r, e);
    }
}

use crate::{
    core_arch::{simd::*, x86::*},
    intrinsics::simd::*,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandw))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm512_mask_expandloadu_epi16(
    src: __m512i,
    k: __mmask32,
    mem_addr: *const i16,
) -> __m512i {
    transmute(expandloadw_512(mem_addr, src.as_i16x32(), k))
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandw))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm512_maskz_expandloadu_epi16(k: __mmask32, mem_addr: *const i16) -> __m512i {
    _mm512_mask_expandloadu_epi16(_mm512_setzero_si512(), k, mem_addr)
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm256_mask_expandloadu_epi16(
    src: __m256i,
    k: __mmask16,
    mem_addr: *const i16,
) -> __m256i {
    transmute(expandloadw_256(mem_addr, src.as_i16x16(), k))
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm256_maskz_expandloadu_epi16(k: __mmask16, mem_addr: *const i16) -> __m256i {
    _mm256_mask_expandloadu_epi16(_mm256_setzero_si256(), k, mem_addr)
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm_mask_expandloadu_epi16(
    src: __m128i,
    k: __mmask8,
    mem_addr: *const i16,
) -> __m128i {
    transmute(expandloadw_128(mem_addr, src.as_i16x8(), k))
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm_maskz_expandloadu_epi16(k: __mmask8, mem_addr: *const i16) -> __m128i {
    _mm_mask_expandloadu_epi16(_mm_setzero_si128(), k, mem_addr)
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandb))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm512_mask_expandloadu_epi8(
    src: __m512i,
    k: __mmask64,
    mem_addr: *const i8,
) -> __m512i {
    transmute(expandloadb_512(mem_addr, src.as_i8x64(), k))
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandb))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm512_maskz_expandloadu_epi8(k: __mmask64, mem_addr: *const i8) -> __m512i {
    _mm512_mask_expandloadu_epi8(_mm512_setzero_si512(), k, mem_addr)
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm256_mask_expandloadu_epi8(
    src: __m256i,
    k: __mmask32,
    mem_addr: *const i8,
) -> __m256i {
    transmute(expandloadb_256(mem_addr, src.as_i8x32(), k))
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm256_maskz_expandloadu_epi8(k: __mmask32, mem_addr: *const i8) -> __m256i {
    _mm256_mask_expandloadu_epi8(_mm256_setzero_si256(), k, mem_addr)
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm_mask_expandloadu_epi8(
    src: __m128i,
    k: __mmask16,
    mem_addr: *const i8,
) -> __m128i {
    transmute(expandloadb_128(mem_addr, src.as_i8x16(), k))
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub unsafe fn _mm_maskz_expandloadu_epi8(k: __mmask16, mem_addr: *const i8) -> __m128i {
    _mm_mask_expandloadu_epi8(_mm_setzero_si128(), k, mem_addr)
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm512_mask_compressstoreu_epi16(base_addr: *mut i16, k: __mmask32, a: __m512i) {
    vcompressstorew(base_addr as *mut _, a.as_i16x32(), k)
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm256_mask_compressstoreu_epi16(base_addr: *mut i16, k: __mmask16, a: __m256i) {
    vcompressstorew256(base_addr as *mut _, a.as_i16x16(), k)
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm_mask_compressstoreu_epi16(base_addr: *mut i16, k: __mmask8, a: __m128i) {
    vcompressstorew128(base_addr as *mut _, a.as_i16x8(), k)
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm512_mask_compressstoreu_epi8(base_addr: *mut i8, k: __mmask64, a: __m512i) {
    vcompressstoreb(base_addr, a.as_i8x64(), k)
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm256_mask_compressstoreu_epi8(base_addr: *mut i8, k: __mmask32, a: __m256i) {
    vcompressstoreb256(base_addr, a.as_i8x32(), k)
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm_mask_compressstoreu_epi8(base_addr: *mut i8, k: __mmask16, a: __m128i) {
    vcompressstoreb128(base_addr, a.as_i8x16(), k)
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compress_epi16&expand=1192)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub fn _mm512_mask_compress_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe { transmute(vpcompressw(a.as_i16x32(), src.as_i16x32(), k)) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_compress_epi16&expand=1193)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub fn _mm512_maskz_compress_epi16(k: __mmask32, a: __m512i) -> __m512i {
    unsafe { transmute(vpcompressw(a.as_i16x32(), i16x32::ZERO, k)) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compress_epi16&expand=1190)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub fn _mm256_mask_compress_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe { transmute(vpcompressw256(a.as_i16x16(), src.as_i16x16(), k)) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_compress_epi16&expand=1191)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub fn _mm256_maskz_compress_epi16(k: __mmask16, a: __m256i) -> __m256i {
    unsafe { transmute(vpcompressw256(a.as_i16x16(), i16x16::ZERO, k)) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compress_epi16&expand=1188)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub fn _mm_mask_compress_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpcompressw128(a.as_i16x8(), src.as_i16x8(), k)) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_compress_epi16&expand=1189)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub fn _mm_maskz_compress_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpcompressw128(a.as_i16x8(), i16x8::ZERO, k)) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compress_epi8&expand=1210)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub fn _mm512_mask_compress_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    unsafe { transmute(vpcompressb(a.as_i8x64(), src.as_i8x64(), k)) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_compress_epi8&expand=1211)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub fn _mm512_maskz_compress_epi8(k: __mmask64, a: __m512i) -> __m512i {
    unsafe { transmute(vpcompressb(a.as_i8x64(), i8x64::ZERO, k)) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compress_epi8&expand=1208)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub fn _mm256_mask_compress_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    unsafe { transmute(vpcompressb256(a.as_i8x32(), src.as_i8x32(), k)) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_compress_epi8&expand=1209)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub fn _mm256_maskz_compress_epi8(k: __mmask32, a: __m256i) -> __m256i {
    unsafe { transmute(vpcompressb256(a.as_i8x32(), i8x32::ZERO, k)) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compress_epi8&expand=1206)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub fn _mm_mask_compress_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    unsafe { transmute(vpcompressb128(a.as_i8x16(), src.as_i8x16(), k)) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_compress_epi8&expand=1207)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub fn _mm_maskz_compress_epi8(k: __mmask16, a: __m128i) -> __m128i {
    unsafe { transmute(vpcompressb128(a.as_i8x16(), i8x16::ZERO, k)) }
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expand_epi16&expand=2310)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub fn _mm512_mask_expand_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe { transmute(vpexpandw(a.as_i16x32(), src.as_i16x32(), k)) }
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expand_epi16&expand=2311)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub fn _mm512_maskz_expand_epi16(k: __mmask32, a: __m512i) -> __m512i {
    unsafe { transmute(vpexpandw(a.as_i16x32(), i16x32::ZERO, k)) }
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expand_epi16&expand=2308)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub fn _mm256_mask_expand_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe { transmute(vpexpandw256(a.as_i16x16(), src.as_i16x16(), k)) }
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expand_epi16&expand=2309)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub fn _mm256_maskz_expand_epi16(k: __mmask16, a: __m256i) -> __m256i {
    unsafe { transmute(vpexpandw256(a.as_i16x16(), i16x16::ZERO, k)) }
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expand_epi16&expand=2306)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub fn _mm_mask_expand_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpexpandw128(a.as_i16x8(), src.as_i16x8(), k)) }
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expand_epi16&expand=2307)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub fn _mm_maskz_expand_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpexpandw128(a.as_i16x8(), i16x8::ZERO, k)) }
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expand_epi8&expand=2328)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub fn _mm512_mask_expand_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    unsafe { transmute(vpexpandb(a.as_i8x64(), src.as_i8x64(), k)) }
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expand_epi8&expand=2329)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub fn _mm512_maskz_expand_epi8(k: __mmask64, a: __m512i) -> __m512i {
    unsafe { transmute(vpexpandb(a.as_i8x64(), i8x64::ZERO, k)) }
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expand_epi8&expand=2326)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub fn _mm256_mask_expand_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    unsafe { transmute(vpexpandb256(a.as_i8x32(), src.as_i8x32(), k)) }
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expand_epi8&expand=2327)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub fn _mm256_maskz_expand_epi8(k: __mmask32, a: __m256i) -> __m256i {
    unsafe { transmute(vpexpandb256(a.as_i8x32(), i8x32::ZERO, k)) }
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expand_epi8&expand=2324)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub fn _mm_mask_expand_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    unsafe { transmute(vpexpandb128(a.as_i8x16(), src.as_i8x16(), k)) }
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expand_epi8&expand=2325)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub fn _mm_maskz_expand_epi8(k: __mmask16, a: __m128i) -> __m128i {
    unsafe { transmute(vpexpandb128(a.as_i8x16(), i8x16::ZERO, k)) }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shldv_epi64&expand=5087)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm512_shldv_epi64(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe { transmute(vpshldvq(a.as_i64x8(), b.as_i64x8(), c.as_i64x8())) }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shldv_epi64&expand=5085)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm512_mask_shldv_epi64(a: __m512i, k: __mmask8, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shldv_epi64(a, b, c).as_i64x8();
        transmute(simd_select_bitmask(k, shf, a.as_i64x8()))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shldv_epi64&expand=5086)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm512_maskz_shldv_epi64(k: __mmask8, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shldv_epi64(a, b, c).as_i64x8();
        transmute(simd_select_bitmask(k, shf, i64x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shldv_epi64&expand=5084)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm256_shldv_epi64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { transmute(vpshldvq256(a.as_i64x4(), b.as_i64x4(), c.as_i64x4())) }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shldv_epi64&expand=5082)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm256_mask_shldv_epi64(a: __m256i, k: __mmask8, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shldv_epi64(a, b, c).as_i64x4();
        transmute(simd_select_bitmask(k, shf, a.as_i64x4()))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shldv_epi64&expand=5083)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm256_maskz_shldv_epi64(k: __mmask8, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shldv_epi64(a, b, c).as_i64x4();
        transmute(simd_select_bitmask(k, shf, i64x4::ZERO))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shldv_epi64&expand=5081)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm_shldv_epi64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe { transmute(vpshldvq128(a.as_i64x2(), b.as_i64x2(), c.as_i64x2())) }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shldv_epi64&expand=5079)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm_mask_shldv_epi64(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shldv_epi64(a, b, c).as_i64x2();
        transmute(simd_select_bitmask(k, shf, a.as_i64x2()))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shldv_epi64&expand=5080)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvq))]
pub fn _mm_maskz_shldv_epi64(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shldv_epi64(a, b, c).as_i64x2();
        transmute(simd_select_bitmask(k, shf, i64x2::ZERO))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shldv_epi32&expand=5078)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm512_shldv_epi32(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe { transmute(vpshldvd(a.as_i32x16(), b.as_i32x16(), c.as_i32x16())) }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shldv_epi32&expand=5076)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm512_mask_shldv_epi32(a: __m512i, k: __mmask16, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shldv_epi32(a, b, c).as_i32x16();
        transmute(simd_select_bitmask(k, shf, a.as_i32x16()))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shldv_epi32&expand=5077)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm512_maskz_shldv_epi32(k: __mmask16, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shldv_epi32(a, b, c).as_i32x16();
        transmute(simd_select_bitmask(k, shf, i32x16::ZERO))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shldv_epi32&expand=5075)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm256_shldv_epi32(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { transmute(vpshldvd256(a.as_i32x8(), b.as_i32x8(), c.as_i32x8())) }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shldv_epi32&expand=5073)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm256_mask_shldv_epi32(a: __m256i, k: __mmask8, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shldv_epi32(a, b, c).as_i32x8();
        transmute(simd_select_bitmask(k, shf, a.as_i32x8()))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shldv_epi32&expand=5074)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm256_maskz_shldv_epi32(k: __mmask8, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shldv_epi32(a, b, c).as_i32x8();
        transmute(simd_select_bitmask(k, shf, i32x8::ZERO))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shldv_epi32&expand=5072)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm_shldv_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe { transmute(vpshldvd128(a.as_i32x4(), b.as_i32x4(), c.as_i32x4())) }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shldv_epi32&expand=5070)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm_mask_shldv_epi32(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shldv_epi32(a, b, c).as_i32x4();
        transmute(simd_select_bitmask(k, shf, a.as_i32x4()))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shldv_epi32&expand=5071)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvd))]
pub fn _mm_maskz_shldv_epi32(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shldv_epi32(a, b, c).as_i32x4();
        transmute(simd_select_bitmask(k, shf, i32x4::ZERO))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shldv_epi16&expand=5069)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm512_shldv_epi16(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe { transmute(vpshldvw(a.as_i16x32(), b.as_i16x32(), c.as_i16x32())) }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shldv_epi16&expand=5067)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm512_mask_shldv_epi16(a: __m512i, k: __mmask32, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shldv_epi16(a, b, c).as_i16x32();
        transmute(simd_select_bitmask(k, shf, a.as_i16x32()))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shldv_epi16&expand=5068)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm512_maskz_shldv_epi16(k: __mmask32, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shldv_epi16(a, b, c).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shldv_epi16&expand=5066)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm256_shldv_epi16(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { transmute(vpshldvw256(a.as_i16x16(), b.as_i16x16(), c.as_i16x16())) }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shldv_epi16&expand=5064)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm256_mask_shldv_epi16(a: __m256i, k: __mmask16, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shldv_epi16(a, b, c).as_i16x16();
        transmute(simd_select_bitmask(k, shf, a.as_i16x16()))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shldv_epi16&expand=5065)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm256_maskz_shldv_epi16(k: __mmask16, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shldv_epi16(a, b, c).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shldv_epi16&expand=5063)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm_shldv_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe { transmute(vpshldvw128(a.as_i16x8(), b.as_i16x8(), c.as_i16x8())) }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shldv_epi16&expand=5061)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm_mask_shldv_epi16(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shldv_epi16(a, b, c).as_i16x8();
        transmute(simd_select_bitmask(k, shf, a.as_i16x8()))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by the amount specified in the corresponding element of c, and store the upper 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shldv_epi16&expand=5062)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldvw))]
pub fn _mm_maskz_shldv_epi16(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shldv_epi16(a, b, c).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shrdv_epi64&expand=5141)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm512_shrdv_epi64(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe { transmute(vpshrdvq(b.as_i64x8(), a.as_i64x8(), c.as_i64x8())) }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shrdv_epi64&expand=5139)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm512_mask_shrdv_epi64(a: __m512i, k: __mmask8, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shrdv_epi64(a, b, c).as_i64x8();
        transmute(simd_select_bitmask(k, shf, a.as_i64x8()))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shrdv_epi64&expand=5140)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm512_maskz_shrdv_epi64(k: __mmask8, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shrdv_epi64(a, b, c).as_i64x8();
        transmute(simd_select_bitmask(k, shf, i64x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shrdv_epi64&expand=5138)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm256_shrdv_epi64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { transmute(vpshrdvq256(b.as_i64x4(), a.as_i64x4(), c.as_i64x4())) }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shrdv_epi64&expand=5136)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm256_mask_shrdv_epi64(a: __m256i, k: __mmask8, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shrdv_epi64(a, b, c).as_i64x4();
        transmute(simd_select_bitmask(k, shf, a.as_i64x4()))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shrdv_epi64&expand=5137)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm256_maskz_shrdv_epi64(k: __mmask8, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shrdv_epi64(a, b, c).as_i64x4();
        transmute(simd_select_bitmask(k, shf, i64x4::ZERO))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shrdv_epi64&expand=5135)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm_shrdv_epi64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe { transmute(vpshrdvq128(b.as_i64x2(), a.as_i64x2(), c.as_i64x2())) }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shrdv_epi64&expand=5133)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm_mask_shrdv_epi64(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shrdv_epi64(a, b, c).as_i64x2();
        transmute(simd_select_bitmask(k, shf, a.as_i64x2()))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shrdv_epi64&expand=5134)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvq))]
pub fn _mm_maskz_shrdv_epi64(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shrdv_epi64(a, b, c).as_i64x2();
        transmute(simd_select_bitmask(k, shf, i64x2::ZERO))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shrdv_epi32&expand=5132)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm512_shrdv_epi32(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe { transmute(vpshrdvd(b.as_i32x16(), a.as_i32x16(), c.as_i32x16())) }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shrdv_epi32&expand=5130)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm512_mask_shrdv_epi32(a: __m512i, k: __mmask16, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shrdv_epi32(a, b, c).as_i32x16();
        transmute(simd_select_bitmask(k, shf, a.as_i32x16()))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shrdv_epi32&expand=5131)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm512_maskz_shrdv_epi32(k: __mmask16, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shrdv_epi32(a, b, c).as_i32x16();
        transmute(simd_select_bitmask(k, shf, i32x16::ZERO))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shrdv_epi32&expand=5129)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm256_shrdv_epi32(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { transmute(vpshrdvd256(b.as_i32x8(), a.as_i32x8(), c.as_i32x8())) }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shrdv_epi32&expand=5127)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm256_mask_shrdv_epi32(a: __m256i, k: __mmask8, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shrdv_epi32(a, b, c).as_i32x8();
        transmute(simd_select_bitmask(k, shf, a.as_i32x8()))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shrdv_epi32&expand=5128)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm256_maskz_shrdv_epi32(k: __mmask8, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shrdv_epi32(a, b, c).as_i32x8();
        transmute(simd_select_bitmask(k, shf, i32x8::ZERO))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shrdv_epi32&expand=5126)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm_shrdv_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe { transmute(vpshrdvd128(b.as_i32x4(), a.as_i32x4(), c.as_i32x4())) }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shrdv_epi32&expand=5124)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm_mask_shrdv_epi32(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shrdv_epi32(a, b, c).as_i32x4();
        transmute(simd_select_bitmask(k, shf, a.as_i32x4()))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shrdv_epi32&expand=5125)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvd))]
pub fn _mm_maskz_shrdv_epi32(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shrdv_epi32(a, b, c).as_i32x4();
        transmute(simd_select_bitmask(k, shf, i32x4::ZERO))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shrdv_epi16&expand=5123)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm512_shrdv_epi16(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe { transmute(vpshrdvw(b.as_i16x32(), a.as_i16x32(), c.as_i16x32())) }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shrdv_epi16&expand=5121)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm512_mask_shrdv_epi16(a: __m512i, k: __mmask32, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shrdv_epi16(a, b, c).as_i16x32();
        transmute(simd_select_bitmask(k, shf, a.as_i16x32()))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shrdv_epi16&expand=5122)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm512_maskz_shrdv_epi16(k: __mmask32, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_shrdv_epi16(a, b, c).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shrdv_epi16&expand=5120)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm256_shrdv_epi16(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { transmute(vpshrdvw256(b.as_i16x16(), a.as_i16x16(), c.as_i16x16())) }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shrdv_epi16&expand=5118)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm256_mask_shrdv_epi16(a: __m256i, k: __mmask16, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shrdv_epi16(a, b, c).as_i16x16();
        transmute(simd_select_bitmask(k, shf, a.as_i16x16()))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shrdv_epi16&expand=5119)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm256_maskz_shrdv_epi16(k: __mmask16, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_shrdv_epi16(a, b, c).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shrdv_epi16&expand=5117)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm_shrdv_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe { transmute(vpshrdvw128(b.as_i16x8(), a.as_i16x8(), c.as_i16x8())) }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shrdv_epi16&expand=5115)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm_mask_shrdv_epi16(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shrdv_epi16(a, b, c).as_i16x8();
        transmute(simd_select_bitmask(k, shf, a.as_i16x8()))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by the amount specified in the corresponding element of c, and store the lower 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shrdv_epi16&expand=5116)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshrdvw))]
pub fn _mm_maskz_shrdv_epi16(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_shrdv_epi16(a, b, c).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shldi_epi64&expand=5060)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_shldi_epi64<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_shldv_epi64(a, b, _mm512_set1_epi64(IMM8 as i64))
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shldi_epi64&expand=5058)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_shldi_epi64<const IMM8: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shldi_epi64::<IMM8>(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shldi_epi64&expand=5059)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_shldi_epi64<const IMM8: i32>(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shldi_epi64::<IMM8>(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, shf, i64x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shldi_epi64&expand=5057)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_shldi_epi64<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_shldv_epi64(a, b, _mm256_set1_epi64x(IMM8 as i64))
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shldi_epi64&expand=5055)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm256_mask_shldi_epi64<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shldi_epi64::<IMM8>(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, shf, src.as_i64x4()))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shldi_epi64&expand=5056)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_maskz_shldi_epi64<const IMM8: i32>(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shldi_epi64::<IMM8>(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, shf, i64x4::ZERO))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shldi_epi64&expand=5054)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_shldi_epi64<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_shldv_epi64(a, b, _mm_set1_epi64x(IMM8 as i64))
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shldi_epi64&expand=5052)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm_mask_shldi_epi64<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shldi_epi64::<IMM8>(a, b).as_i64x2();
        transmute(simd_select_bitmask(k, shf, src.as_i64x2()))
    }
}

/// Concatenate packed 64-bit integers in a and b producing an intermediate 128-bit result. Shift the result left by imm8 bits, and store the upper 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shldi_epi64&expand=5053)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_maskz_shldi_epi64<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shldi_epi64::<IMM8>(a, b).as_i64x2();
        transmute(simd_select_bitmask(k, shf, i64x2::ZERO))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shldi_epi32&expand=5051)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_shldi_epi32<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_shldv_epi32(a, b, _mm512_set1_epi32(IMM8))
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shldi_epi32&expand=5049)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_shldi_epi32<const IMM8: i32>(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shldi_epi32::<IMM8>(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shldi_epi32&expand=5050)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_shldi_epi32<const IMM8: i32>(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shldi_epi32::<IMM8>(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, shf, i32x16::ZERO))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shldi_epi32&expand=5048)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_shldi_epi32<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_shldv_epi32(a, b, _mm256_set1_epi32(IMM8))
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shldi_epi32&expand=5046)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm256_mask_shldi_epi32<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shldi_epi32::<IMM8>(a, b).as_i32x8();
        transmute(simd_select_bitmask(k, shf, src.as_i32x8()))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shldi_epi32&expand=5047)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_maskz_shldi_epi32<const IMM8: i32>(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shldi_epi32::<IMM8>(a, b).as_i32x8();
        transmute(simd_select_bitmask(k, shf, i32x8::ZERO))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shldi_epi32&expand=5045)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_shldi_epi32<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_shldv_epi32(a, b, _mm_set1_epi32(IMM8))
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shldi_epi32&expand=5043)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm_mask_shldi_epi32<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shldi_epi32::<IMM8>(a, b).as_i32x4();
        transmute(simd_select_bitmask(k, shf, src.as_i32x4()))
    }
}

/// Concatenate packed 32-bit integers in a and b producing an intermediate 64-bit result. Shift the result left by imm8 bits, and store the upper 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shldi_epi32&expand=5044)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_maskz_shldi_epi32<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shldi_epi32::<IMM8>(a, b).as_i32x4();
        transmute(simd_select_bitmask(k, shf, i32x4::ZERO))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shldi_epi16&expand=5042)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_shldi_epi16<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_shldv_epi16(a, b, _mm512_set1_epi16(IMM8 as i16))
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shldi_epi16&expand=5040)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_shldi_epi16<const IMM8: i32>(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shldi_epi16::<IMM8>(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shldi_epi16&expand=5041)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_shldi_epi16<const IMM8: i32>(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shldi_epi16::<IMM8>(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shldi_epi16&expand=5039)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_shldi_epi16<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_shldv_epi16(a, b, _mm256_set1_epi16(IMM8 as i16))
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shldi_epi16&expand=5037)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm256_mask_shldi_epi16<const IMM8: i32>(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shldi_epi16::<IMM8>(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shldi_epi16&expand=5038)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_maskz_shldi_epi16<const IMM8: i32>(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shldi_epi16::<IMM8>(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shldi_epi16&expand=5036)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_shldi_epi16<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_shldv_epi16(a, b, _mm_set1_epi16(IMM8 as i16))
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shldi_epi16&expand=5034)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
pub fn _mm_mask_shldi_epi16<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shldi_epi16::<IMM8>(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Concatenate packed 16-bit integers in a and b producing an intermediate 32-bit result. Shift the result left by imm8 bits, and store the upper 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shldi_epi16&expand=5035)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_maskz_shldi_epi16<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shldi_epi16::<IMM8>(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shrdi_epi64&expand=5114)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(2)]
pub fn _mm512_shrdi_epi64<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_shrdv_epi64(a, b, _mm512_set1_epi64(IMM8 as i64))
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst using writemask k (elements are copied from src" when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shrdi_epi64&expand=5112)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_shrdi_epi64<const IMM8: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shrdi_epi64::<IMM8>(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, shf, src.as_i64x8()))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shrdi_epi64&expand=5113)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 255))] //should be vpshrdq
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_shrdi_epi64<const IMM8: i32>(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shrdi_epi64::<IMM8>(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, shf, i64x8::ZERO))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shrdi_epi64&expand=5111)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(2)]
pub fn _mm256_shrdi_epi64<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_shrdv_epi64(a, b, _mm256_set1_epi64x(IMM8 as i64))
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst using writemask k (elements are copied from src" when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shrdi_epi64&expand=5109)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(4)]
pub fn _mm256_mask_shrdi_epi64<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shrdi_epi64::<IMM8>(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, shf, src.as_i64x4()))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shrdi_epi64&expand=5110)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(3)]
pub fn _mm256_maskz_shrdi_epi64<const IMM8: i32>(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shrdi_epi64::<IMM8>(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, shf, i64x4::ZERO))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shrdi_epi64&expand=5108)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(2)]
pub fn _mm_shrdi_epi64<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_shrdv_epi64(a, b, _mm_set1_epi64x(IMM8 as i64))
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst using writemask k (elements are copied from src" when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shrdi_epi64&expand=5106)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(4)]
pub fn _mm_mask_shrdi_epi64<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shrdi_epi64::<IMM8>(a, b).as_i64x2();
        transmute(simd_select_bitmask(k, shf, src.as_i64x2()))
    }
}

/// Concatenate packed 64-bit integers in b and a producing an intermediate 128-bit result. Shift the result right by imm8 bits, and store the lower 64-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shrdi_epi64&expand=5107)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldq, IMM8 = 5))] //should be vpshrdq
#[rustc_legacy_const_generics(3)]
pub fn _mm_maskz_shrdi_epi64<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shrdi_epi64::<IMM8>(a, b).as_i64x2();
        transmute(simd_select_bitmask(k, shf, i64x2::ZERO))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shrdi_epi32&expand=5105)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(2)]
pub fn _mm512_shrdi_epi32<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_shrdv_epi32(a, b, _mm512_set1_epi32(IMM8))
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shrdi_epi32&expand=5103)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_shrdi_epi32<const IMM8: i32>(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shrdi_epi32::<IMM8>(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shrdi_epi32&expand=5104)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_shrdi_epi32<const IMM8: i32>(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shrdi_epi32::<IMM8>(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, shf, i32x16::ZERO))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shrdi_epi32&expand=5102)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(2)]
pub fn _mm256_shrdi_epi32<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_shrdv_epi32(a, b, _mm256_set1_epi32(IMM8))
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shrdi_epi32&expand=5100)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(4)]
pub fn _mm256_mask_shrdi_epi32<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shrdi_epi32::<IMM8>(a, b).as_i32x8();
        transmute(simd_select_bitmask(k, shf, src.as_i32x8()))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shrdi_epi32&expand=5101)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(3)]
pub fn _mm256_maskz_shrdi_epi32<const IMM8: i32>(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shrdi_epi32::<IMM8>(a, b).as_i32x8();
        transmute(simd_select_bitmask(k, shf, i32x8::ZERO))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shrdi_epi32&expand=5099)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(2)]
pub fn _mm_shrdi_epi32<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_shrdv_epi32(a, b, _mm_set1_epi32(IMM8))
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shrdi_epi32&expand=5097)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(4)]
pub fn _mm_mask_shrdi_epi32<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shrdi_epi32::<IMM8>(a, b).as_i32x4();
        transmute(simd_select_bitmask(k, shf, src.as_i32x4()))
    }
}

/// Concatenate packed 32-bit integers in b and a producing an intermediate 64-bit result. Shift the result right by imm8 bits, and store the lower 32-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shrdi_epi32&expand=5098)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldd, IMM8 = 5))] //should be vpshldd
#[rustc_legacy_const_generics(3)]
pub fn _mm_maskz_shrdi_epi32<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shrdi_epi32::<IMM8>(a, b).as_i32x4();
        transmute(simd_select_bitmask(k, shf, i32x4::ZERO))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shrdi_epi16&expand=5096)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(2)]
pub fn _mm512_shrdi_epi16<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_shrdv_epi16(a, b, _mm512_set1_epi16(IMM8 as i16))
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shrdi_epi16&expand=5094)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_shrdi_epi16<const IMM8: i32>(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shrdi_epi16::<IMM8>(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shrdi_epi16&expand=5095)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_shrdi_epi16<const IMM8: i32>(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm512_shrdi_epi16::<IMM8>(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shrdi_epi16&expand=5093)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(2)]
pub fn _mm256_shrdi_epi16<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_shrdv_epi16(a, b, _mm256_set1_epi16(IMM8 as i16))
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shrdi_epi16&expand=5091)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(4)]
pub fn _mm256_mask_shrdi_epi16<const IMM8: i32>(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shrdi_epi16::<IMM8>(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shrdi_epi16&expand=5092)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(3)]
pub fn _mm256_maskz_shrdi_epi16<const IMM8: i32>(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_shrdi_epi16::<IMM8>(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shrdi_epi16&expand=5090)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(2)]
pub fn _mm_shrdi_epi16<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_shrdv_epi16(a, b, _mm_set1_epi16(IMM8 as i16))
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shrdi_epi16&expand=5088)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(4)]
pub fn _mm_mask_shrdi_epi16<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shrdi_epi16::<IMM8>(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Concatenate packed 16-bit integers in b and a producing an intermediate 32-bit result. Shift the result right by imm8 bits, and store the lower 16-bits in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shrdi_epi16&expand=5089)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vpshldw, IMM8 = 5))] //should be vpshrdw
#[rustc_legacy_const_generics(3)]
pub fn _mm_maskz_shrdi_epi16<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_shrdi_epi16::<IMM8>(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512.mask.compress.store.w.512"]
    fn vcompressstorew(mem: *mut i8, data: i16x32, mask: u32);
    #[link_name = "llvm.x86.avx512.mask.compress.store.w.256"]
    fn vcompressstorew256(mem: *mut i8, data: i16x16, mask: u16);
    #[link_name = "llvm.x86.avx512.mask.compress.store.w.128"]
    fn vcompressstorew128(mem: *mut i8, data: i16x8, mask: u8);

    #[link_name = "llvm.x86.avx512.mask.compress.store.b.512"]
    fn vcompressstoreb(mem: *mut i8, data: i8x64, mask: u64);
    #[link_name = "llvm.x86.avx512.mask.compress.store.b.256"]
    fn vcompressstoreb256(mem: *mut i8, data: i8x32, mask: u32);
    #[link_name = "llvm.x86.avx512.mask.compress.store.b.128"]
    fn vcompressstoreb128(mem: *mut i8, data: i8x16, mask: u16);

    #[link_name = "llvm.x86.avx512.mask.compress.w.512"]
    fn vpcompressw(a: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.compress.w.256"]
    fn vpcompressw256(a: i16x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.compress.w.128"]
    fn vpcompressw128(a: i16x8, src: i16x8, mask: u8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.compress.b.512"]
    fn vpcompressb(a: i8x64, src: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.compress.b.256"]
    fn vpcompressb256(a: i8x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.compress.b.128"]
    fn vpcompressb128(a: i8x16, src: i8x16, mask: u16) -> i8x16;

    #[link_name = "llvm.x86.avx512.mask.expand.w.512"]
    fn vpexpandw(a: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.expand.w.256"]
    fn vpexpandw256(a: i16x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.expand.w.128"]
    fn vpexpandw128(a: i16x8, src: i16x8, mask: u8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.expand.b.512"]
    fn vpexpandb(a: i8x64, src: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.expand.b.256"]
    fn vpexpandb256(a: i8x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.expand.b.128"]
    fn vpexpandb128(a: i8x16, src: i8x16, mask: u16) -> i8x16;

    #[link_name = "llvm.fshl.v8i64"]
    fn vpshldvq(a: i64x8, b: i64x8, c: i64x8) -> i64x8;
    #[link_name = "llvm.fshl.v4i64"]
    fn vpshldvq256(a: i64x4, b: i64x4, c: i64x4) -> i64x4;
    #[link_name = "llvm.fshl.v2i64"]
    fn vpshldvq128(a: i64x2, b: i64x2, c: i64x2) -> i64x2;
    #[link_name = "llvm.fshl.v16i32"]
    fn vpshldvd(a: i32x16, b: i32x16, c: i32x16) -> i32x16;
    #[link_name = "llvm.fshl.v8i32"]
    fn vpshldvd256(a: i32x8, b: i32x8, c: i32x8) -> i32x8;
    #[link_name = "llvm.fshl.v4i32"]
    fn vpshldvd128(a: i32x4, b: i32x4, c: i32x4) -> i32x4;
    #[link_name = "llvm.fshl.v32i16"]
    fn vpshldvw(a: i16x32, b: i16x32, c: i16x32) -> i16x32;
    #[link_name = "llvm.fshl.v16i16"]
    fn vpshldvw256(a: i16x16, b: i16x16, c: i16x16) -> i16x16;
    #[link_name = "llvm.fshl.v8i16"]
    fn vpshldvw128(a: i16x8, b: i16x8, c: i16x8) -> i16x8;

    #[link_name = "llvm.fshr.v8i64"]
    fn vpshrdvq(a: i64x8, b: i64x8, c: i64x8) -> i64x8;
    #[link_name = "llvm.fshr.v4i64"]
    fn vpshrdvq256(a: i64x4, b: i64x4, c: i64x4) -> i64x4;
    #[link_name = "llvm.fshr.v2i64"]
    fn vpshrdvq128(a: i64x2, b: i64x2, c: i64x2) -> i64x2;
    #[link_name = "llvm.fshr.v16i32"]
    fn vpshrdvd(a: i32x16, b: i32x16, c: i32x16) -> i32x16;
    #[link_name = "llvm.fshr.v8i32"]
    fn vpshrdvd256(a: i32x8, b: i32x8, c: i32x8) -> i32x8;
    #[link_name = "llvm.fshr.v4i32"]
    fn vpshrdvd128(a: i32x4, b: i32x4, c: i32x4) -> i32x4;
    #[link_name = "llvm.fshr.v32i16"]
    fn vpshrdvw(a: i16x32, b: i16x32, c: i16x32) -> i16x32;
    #[link_name = "llvm.fshr.v16i16"]
    fn vpshrdvw256(a: i16x16, b: i16x16, c: i16x16) -> i16x16;
    #[link_name = "llvm.fshr.v8i16"]
    fn vpshrdvw128(a: i16x8, b: i16x8, c: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.expand.load.b.128"]
    fn expandloadb_128(mem_addr: *const i8, a: i8x16, mask: u16) -> i8x16;
    #[link_name = "llvm.x86.avx512.mask.expand.load.w.128"]
    fn expandloadw_128(mem_addr: *const i16, a: i16x8, mask: u8) -> i16x8;
    #[link_name = "llvm.x86.avx512.mask.expand.load.b.256"]
    fn expandloadb_256(mem_addr: *const i8, a: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.expand.load.w.256"]
    fn expandloadw_256(mem_addr: *const i16, a: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.expand.load.b.512"]
    fn expandloadb_512(mem_addr: *const i8, a: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.expand.load.w.512"]
    fn expandloadw_512(mem_addr: *const i16, a: i16x32, mask: u32) -> i16x32;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::hint::black_box;

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_compress_epi16() {
        let src = _mm512_set1_epi16(200);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_mask_compress_epi16(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
            1,   3,   5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_compress_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_maskz_compress_epi16(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_compress_epi16() {
        let src = _mm256_set1_epi16(200);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_compress_epi16(src, 0b01010101_01010101, a);
        let e = _mm256_set_epi16(
            200, 200, 200, 200, 200, 200, 200, 200, 1, 3, 5, 7, 9, 11, 13, 15,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_compress_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_compress_epi16(0b01010101_01010101, a);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_compress_epi16() {
        let src = _mm_set1_epi16(200);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_compress_epi16(src, 0b01010101, a);
        let e = _mm_set_epi16(200, 200, 200, 200, 1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_compress_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_compress_epi16(0b01010101, a);
        let e = _mm_set_epi16(0, 0, 0, 0, 1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_compress_epi8() {
        let src = _mm512_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_mask_compress_epi8(
            src,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            1,   3,   5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
            33,  35,  37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_compress_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_maskz_compress_epi8(
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_compress_epi8() {
        let src = _mm256_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_mask_compress_epi8(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            1,   3,   5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_compress_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_maskz_compress_epi8(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_compress_epi8() {
        let src = _mm_set1_epi8(100);
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_mask_compress_epi8(src, 0b01010101_01010101, a);
        let e = _mm_set_epi8(
            100, 100, 100, 100, 100, 100, 100, 100, 1, 3, 5, 7, 9, 11, 13, 15,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_compress_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_maskz_compress_epi8(0b01010101_01010101, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_expand_epi16() {
        let src = _mm512_set1_epi16(200);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_mask_expand_epi16(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            200, 16, 200, 17, 200, 18, 200, 19, 200, 20, 200, 21, 200, 22, 200, 23,
            200, 24, 200, 25, 200, 26, 200, 27, 200, 28, 200, 29, 200, 30, 200, 31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_expand_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_maskz_expand_epi16(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23,
                                 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_expand_epi16() {
        let src = _mm256_set1_epi16(200);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_expand_epi16(src, 0b01010101_01010101, a);
        let e = _mm256_set_epi16(
            200, 8, 200, 9, 200, 10, 200, 11, 200, 12, 200, 13, 200, 14, 200, 15,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_expand_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_expand_epi16(0b01010101_01010101, a);
        let e = _mm256_set_epi16(0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_expand_epi16() {
        let src = _mm_set1_epi16(200);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_expand_epi16(src, 0b01010101, a);
        let e = _mm_set_epi16(200, 4, 200, 5, 200, 6, 200, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_expand_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_expand_epi16(0b01010101, a);
        let e = _mm_set_epi16(0, 4, 0, 5, 0, 6, 0, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_expand_epi8() {
        let src = _mm512_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_mask_expand_epi8(
            src,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            100, 32, 100, 33, 100, 34, 100, 35, 100, 36, 100, 37, 100, 38, 100, 39,
            100, 40, 100, 41, 100, 42, 100, 43, 100, 44, 100, 45, 100, 46, 100, 47,
            100, 48, 100, 49, 100, 50, 100, 51, 100, 52, 100, 53, 100, 54, 100, 55,
            100, 56, 100, 57, 100, 58, 100, 59, 100, 60, 100, 61, 100, 62, 100, 63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_expand_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_maskz_expand_epi8(
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39,
            0, 40, 0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47,
            0, 48, 0, 49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55,
            0, 56, 0, 57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_expand_epi8() {
        let src = _mm256_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_mask_expand_epi8(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            100, 16, 100, 17, 100, 18, 100, 19, 100, 20, 100, 21, 100, 22, 100, 23,
            100, 24, 100, 25, 100, 26, 100, 27, 100, 28, 100, 29, 100, 30, 100, 31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_expand_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_maskz_expand_epi8(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23,
            0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_expand_epi8() {
        let src = _mm_set1_epi8(100);
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_mask_expand_epi8(src, 0b01010101_01010101, a);
        let e = _mm_set_epi8(
            100, 8, 100, 9, 100, 10, 100, 11, 100, 12, 100, 13, 100, 14, 100, 15,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_expand_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_maskz_expand_epi8(0b01010101_01010101, a);
        let e = _mm_set_epi8(0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shldv_epi64() {
        let a = _mm512_set1_epi64(1);
        let b = _mm512_set1_epi64(1 << 63);
        let c = _mm512_set1_epi64(2);
        let r = _mm512_shldv_epi64(a, b, c);
        let e = _mm512_set1_epi64(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shldv_epi64() {
        let a = _mm512_set1_epi64(1);
        let b = _mm512_set1_epi64(1 << 63);
        let c = _mm512_set1_epi64(2);
        let r = _mm512_mask_shldv_epi64(a, 0, b, c);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shldv_epi64(a, 0b11111111, b, c);
        let e = _mm512_set1_epi64(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shldv_epi64() {
        let a = _mm512_set1_epi64(1);
        let b = _mm512_set1_epi64(1 << 63);
        let c = _mm512_set1_epi64(2);
        let r = _mm512_maskz_shldv_epi64(0, a, b, c);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shldv_epi64(0b11111111, a, b, c);
        let e = _mm512_set1_epi64(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shldv_epi64() {
        let a = _mm256_set1_epi64x(1);
        let b = _mm256_set1_epi64x(1 << 63);
        let c = _mm256_set1_epi64x(2);
        let r = _mm256_shldv_epi64(a, b, c);
        let e = _mm256_set1_epi64x(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shldv_epi64() {
        let a = _mm256_set1_epi64x(1);
        let b = _mm256_set1_epi64x(1 << 63);
        let c = _mm256_set1_epi64x(2);
        let r = _mm256_mask_shldv_epi64(a, 0, b, c);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shldv_epi64(a, 0b00001111, b, c);
        let e = _mm256_set1_epi64x(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shldv_epi64() {
        let a = _mm256_set1_epi64x(1);
        let b = _mm256_set1_epi64x(1 << 63);
        let c = _mm256_set1_epi64x(2);
        let r = _mm256_maskz_shldv_epi64(0, a, b, c);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shldv_epi64(0b00001111, a, b, c);
        let e = _mm256_set1_epi64x(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shldv_epi64() {
        let a = _mm_set1_epi64x(1);
        let b = _mm_set1_epi64x(1 << 63);
        let c = _mm_set1_epi64x(2);
        let r = _mm_shldv_epi64(a, b, c);
        let e = _mm_set1_epi64x(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shldv_epi64() {
        let a = _mm_set1_epi64x(1);
        let b = _mm_set1_epi64x(1 << 63);
        let c = _mm_set1_epi64x(2);
        let r = _mm_mask_shldv_epi64(a, 0, b, c);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shldv_epi64(a, 0b00000011, b, c);
        let e = _mm_set1_epi64x(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shldv_epi64() {
        let a = _mm_set1_epi64x(1);
        let b = _mm_set1_epi64x(1 << 63);
        let c = _mm_set1_epi64x(2);
        let r = _mm_maskz_shldv_epi64(0, a, b, c);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shldv_epi64(0b00000011, a, b, c);
        let e = _mm_set1_epi64x(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shldv_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1 << 31);
        let c = _mm512_set1_epi32(2);
        let r = _mm512_shldv_epi32(a, b, c);
        let e = _mm512_set1_epi32(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shldv_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1 << 31);
        let c = _mm512_set1_epi32(2);
        let r = _mm512_mask_shldv_epi32(a, 0, b, c);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shldv_epi32(a, 0b11111111_11111111, b, c);
        let e = _mm512_set1_epi32(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shldv_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1 << 31);
        let c = _mm512_set1_epi32(2);
        let r = _mm512_maskz_shldv_epi32(0, a, b, c);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shldv_epi32(0b11111111_11111111, a, b, c);
        let e = _mm512_set1_epi32(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shldv_epi32() {
        let a = _mm256_set1_epi32(1);
        let b = _mm256_set1_epi32(1 << 31);
        let c = _mm256_set1_epi32(2);
        let r = _mm256_shldv_epi32(a, b, c);
        let e = _mm256_set1_epi32(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shldv_epi32() {
        let a = _mm256_set1_epi32(1);
        let b = _mm256_set1_epi32(1 << 31);
        let c = _mm256_set1_epi32(2);
        let r = _mm256_mask_shldv_epi32(a, 0, b, c);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shldv_epi32(a, 0b11111111, b, c);
        let e = _mm256_set1_epi32(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shldv_epi32() {
        let a = _mm256_set1_epi32(1);
        let b = _mm256_set1_epi32(1 << 31);
        let c = _mm256_set1_epi32(2);
        let r = _mm256_maskz_shldv_epi32(0, a, b, c);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shldv_epi32(0b11111111, a, b, c);
        let e = _mm256_set1_epi32(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shldv_epi32() {
        let a = _mm_set1_epi32(1);
        let b = _mm_set1_epi32(1 << 31);
        let c = _mm_set1_epi32(2);
        let r = _mm_shldv_epi32(a, b, c);
        let e = _mm_set1_epi32(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shldv_epi32() {
        let a = _mm_set1_epi32(1);
        let b = _mm_set1_epi32(1 << 31);
        let c = _mm_set1_epi32(2);
        let r = _mm_mask_shldv_epi32(a, 0, b, c);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shldv_epi32(a, 0b00001111, b, c);
        let e = _mm_set1_epi32(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shldv_epi32() {
        let a = _mm_set1_epi32(1);
        let b = _mm_set1_epi32(1 << 31);
        let c = _mm_set1_epi32(2);
        let r = _mm_maskz_shldv_epi32(0, a, b, c);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shldv_epi32(0b00001111, a, b, c);
        let e = _mm_set1_epi32(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shldv_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1 << 15);
        let c = _mm512_set1_epi16(2);
        let r = _mm512_shldv_epi16(a, b, c);
        let e = _mm512_set1_epi16(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shldv_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1 << 15);
        let c = _mm512_set1_epi16(2);
        let r = _mm512_mask_shldv_epi16(a, 0, b, c);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shldv_epi16(a, 0b11111111_11111111_11111111_11111111, b, c);
        let e = _mm512_set1_epi16(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shldv_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1 << 15);
        let c = _mm512_set1_epi16(2);
        let r = _mm512_maskz_shldv_epi16(0, a, b, c);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shldv_epi16(0b11111111_11111111_11111111_11111111, a, b, c);
        let e = _mm512_set1_epi16(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shldv_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1 << 15);
        let c = _mm256_set1_epi16(2);
        let r = _mm256_shldv_epi16(a, b, c);
        let e = _mm256_set1_epi16(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shldv_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1 << 15);
        let c = _mm256_set1_epi16(2);
        let r = _mm256_mask_shldv_epi16(a, 0, b, c);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shldv_epi16(a, 0b11111111_11111111, b, c);
        let e = _mm256_set1_epi16(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shldv_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1 << 15);
        let c = _mm256_set1_epi16(2);
        let r = _mm256_maskz_shldv_epi16(0, a, b, c);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shldv_epi16(0b11111111_11111111, a, b, c);
        let e = _mm256_set1_epi16(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shldv_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1 << 15);
        let c = _mm_set1_epi16(2);
        let r = _mm_shldv_epi16(a, b, c);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shldv_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1 << 15);
        let c = _mm_set1_epi16(2);
        let r = _mm_mask_shldv_epi16(a, 0, b, c);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shldv_epi16(a, 0b11111111, b, c);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shldv_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1 << 15);
        let c = _mm_set1_epi16(2);
        let r = _mm_maskz_shldv_epi16(0, a, b, c);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shldv_epi16(0b11111111, a, b, c);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shrdv_epi64() {
        let a = _mm512_set1_epi64(2);
        let b = _mm512_set1_epi64(8);
        let c = _mm512_set1_epi64(1);
        let r = _mm512_shrdv_epi64(a, b, c);
        let e = _mm512_set1_epi64(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shrdv_epi64() {
        let a = _mm512_set1_epi64(2);
        let b = _mm512_set1_epi64(8);
        let c = _mm512_set1_epi64(1);
        let r = _mm512_mask_shrdv_epi64(a, 0, b, c);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shrdv_epi64(a, 0b11111111, b, c);
        let e = _mm512_set1_epi64(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shrdv_epi64() {
        let a = _mm512_set1_epi64(2);
        let b = _mm512_set1_epi64(8);
        let c = _mm512_set1_epi64(1);
        let r = _mm512_maskz_shrdv_epi64(0, a, b, c);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shrdv_epi64(0b11111111, a, b, c);
        let e = _mm512_set1_epi64(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shrdv_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(8);
        let c = _mm256_set1_epi64x(1);
        let r = _mm256_shrdv_epi64(a, b, c);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shrdv_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(8);
        let c = _mm256_set1_epi64x(1);
        let r = _mm256_mask_shrdv_epi64(a, 0, b, c);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shrdv_epi64(a, 0b00001111, b, c);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shrdv_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(8);
        let c = _mm256_set1_epi64x(1);
        let r = _mm256_maskz_shrdv_epi64(0, a, b, c);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shrdv_epi64(0b00001111, a, b, c);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shrdv_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(8);
        let c = _mm_set1_epi64x(1);
        let r = _mm_shrdv_epi64(a, b, c);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shrdv_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(8);
        let c = _mm_set1_epi64x(1);
        let r = _mm_mask_shrdv_epi64(a, 0, b, c);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shrdv_epi64(a, 0b00000011, b, c);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shrdv_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(8);
        let c = _mm_set1_epi64x(1);
        let r = _mm_maskz_shrdv_epi64(0, a, b, c);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shrdv_epi64(0b00000011, a, b, c);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shrdv_epi32() {
        let a = _mm512_set1_epi32(2);
        let b = _mm512_set1_epi32(8);
        let c = _mm512_set1_epi32(1);
        let r = _mm512_shrdv_epi32(a, b, c);
        let e = _mm512_set1_epi32(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shrdv_epi32() {
        let a = _mm512_set1_epi32(2);
        let b = _mm512_set1_epi32(8);
        let c = _mm512_set1_epi32(1);
        let r = _mm512_mask_shrdv_epi32(a, 0, b, c);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shrdv_epi32(a, 0b11111111_11111111, b, c);
        let e = _mm512_set1_epi32(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shrdv_epi32() {
        let a = _mm512_set1_epi32(2);
        let b = _mm512_set1_epi32(8);
        let c = _mm512_set1_epi32(1);
        let r = _mm512_maskz_shrdv_epi32(0, a, b, c);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shrdv_epi32(0b11111111_11111111, a, b, c);
        let e = _mm512_set1_epi32(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shrdv_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(8);
        let c = _mm256_set1_epi32(1);
        let r = _mm256_shrdv_epi32(a, b, c);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shrdv_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(8);
        let c = _mm256_set1_epi32(1);
        let r = _mm256_mask_shrdv_epi32(a, 0, b, c);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shrdv_epi32(a, 0b11111111, b, c);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shrdv_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(8);
        let c = _mm256_set1_epi32(1);
        let r = _mm256_maskz_shrdv_epi32(0, a, b, c);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shrdv_epi32(0b11111111, a, b, c);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shrdv_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(8);
        let c = _mm_set1_epi32(1);
        let r = _mm_shrdv_epi32(a, b, c);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shrdv_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(8);
        let c = _mm_set1_epi32(1);
        let r = _mm_mask_shrdv_epi32(a, 0, b, c);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shrdv_epi32(a, 0b00001111, b, c);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shrdv_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(8);
        let c = _mm_set1_epi32(1);
        let r = _mm_maskz_shrdv_epi32(0, a, b, c);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shrdv_epi32(0b00001111, a, b, c);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shrdv_epi16() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(8);
        let c = _mm512_set1_epi16(1);
        let r = _mm512_shrdv_epi16(a, b, c);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shrdv_epi16() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(8);
        let c = _mm512_set1_epi16(1);
        let r = _mm512_mask_shrdv_epi16(a, 0, b, c);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shrdv_epi16(a, 0b11111111_11111111_11111111_11111111, b, c);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shrdv_epi16() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(8);
        let c = _mm512_set1_epi16(1);
        let r = _mm512_maskz_shrdv_epi16(0, a, b, c);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shrdv_epi16(0b11111111_11111111_11111111_11111111, a, b, c);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shrdv_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(8);
        let c = _mm256_set1_epi16(1);
        let r = _mm256_shrdv_epi16(a, b, c);
        let e = _mm256_set1_epi16(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shrdv_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(8);
        let c = _mm256_set1_epi16(1);
        let r = _mm256_mask_shrdv_epi16(a, 0, b, c);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shrdv_epi16(a, 0b11111111_11111111, b, c);
        let e = _mm256_set1_epi16(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shrdv_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(8);
        let c = _mm256_set1_epi16(1);
        let r = _mm256_maskz_shrdv_epi16(0, a, b, c);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shrdv_epi16(0b11111111_11111111, a, b, c);
        let e = _mm256_set1_epi16(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shrdv_epi16() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(8);
        let c = _mm_set1_epi16(1);
        let r = _mm_shrdv_epi16(a, b, c);
        let e = _mm_set1_epi16(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shrdv_epi16() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(8);
        let c = _mm_set1_epi16(1);
        let r = _mm_mask_shrdv_epi16(a, 0, b, c);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shrdv_epi16(a, 0b11111111, b, c);
        let e = _mm_set1_epi16(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shrdv_epi16() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(8);
        let c = _mm_set1_epi16(1);
        let r = _mm_maskz_shrdv_epi16(0, a, b, c);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shrdv_epi16(0b11111111, a, b, c);
        let e = _mm_set1_epi16(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shldi_epi64() {
        let a = _mm512_set1_epi64(1);
        let b = _mm512_set1_epi64(1 << 63);
        let r = _mm512_shldi_epi64::<2>(a, b);
        let e = _mm512_set1_epi64(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shldi_epi64() {
        let a = _mm512_set1_epi64(1);
        let b = _mm512_set1_epi64(1 << 63);
        let r = _mm512_mask_shldi_epi64::<2>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shldi_epi64::<2>(a, 0b11111111, a, b);
        let e = _mm512_set1_epi64(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shldi_epi64() {
        let a = _mm512_set1_epi64(1);
        let b = _mm512_set1_epi64(1 << 63);
        let r = _mm512_maskz_shldi_epi64::<2>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shldi_epi64::<2>(0b11111111, a, b);
        let e = _mm512_set1_epi64(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shldi_epi64() {
        let a = _mm256_set1_epi64x(1);
        let b = _mm256_set1_epi64x(1 << 63);
        let r = _mm256_shldi_epi64::<2>(a, b);
        let e = _mm256_set1_epi64x(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shldi_epi64() {
        let a = _mm256_set1_epi64x(1);
        let b = _mm256_set1_epi64x(1 << 63);
        let r = _mm256_mask_shldi_epi64::<2>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shldi_epi64::<2>(a, 0b00001111, a, b);
        let e = _mm256_set1_epi64x(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shldi_epi64() {
        let a = _mm256_set1_epi64x(1);
        let b = _mm256_set1_epi64x(1 << 63);
        let r = _mm256_maskz_shldi_epi64::<2>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shldi_epi64::<2>(0b00001111, a, b);
        let e = _mm256_set1_epi64x(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shldi_epi64() {
        let a = _mm_set1_epi64x(1);
        let b = _mm_set1_epi64x(1 << 63);
        let r = _mm_shldi_epi64::<2>(a, b);
        let e = _mm_set1_epi64x(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shldi_epi64() {
        let a = _mm_set1_epi64x(1);
        let b = _mm_set1_epi64x(1 << 63);
        let r = _mm_mask_shldi_epi64::<2>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shldi_epi64::<2>(a, 0b00000011, a, b);
        let e = _mm_set1_epi64x(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shldi_epi64() {
        let a = _mm_set1_epi64x(1);
        let b = _mm_set1_epi64x(1 << 63);
        let r = _mm_maskz_shldi_epi64::<2>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shldi_epi64::<2>(0b00000011, a, b);
        let e = _mm_set1_epi64x(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shldi_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1 << 31);
        let r = _mm512_shldi_epi32::<2>(a, b);
        let e = _mm512_set1_epi32(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shldi_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1 << 31);
        let r = _mm512_mask_shldi_epi32::<2>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shldi_epi32::<2>(a, 0b11111111_11111111, a, b);
        let e = _mm512_set1_epi32(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shldi_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1 << 31);
        let r = _mm512_maskz_shldi_epi32::<2>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shldi_epi32::<2>(0b11111111_11111111, a, b);
        let e = _mm512_set1_epi32(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shldi_epi32() {
        let a = _mm256_set1_epi32(1);
        let b = _mm256_set1_epi32(1 << 31);
        let r = _mm256_shldi_epi32::<2>(a, b);
        let e = _mm256_set1_epi32(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shldi_epi32() {
        let a = _mm256_set1_epi32(1);
        let b = _mm256_set1_epi32(1 << 31);
        let r = _mm256_mask_shldi_epi32::<2>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shldi_epi32::<2>(a, 0b11111111, a, b);
        let e = _mm256_set1_epi32(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shldi_epi32() {
        let a = _mm256_set1_epi32(1);
        let b = _mm256_set1_epi32(1 << 31);
        let r = _mm256_maskz_shldi_epi32::<2>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shldi_epi32::<2>(0b11111111, a, b);
        let e = _mm256_set1_epi32(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shldi_epi32() {
        let a = _mm_set1_epi32(1);
        let b = _mm_set1_epi32(1 << 31);
        let r = _mm_shldi_epi32::<2>(a, b);
        let e = _mm_set1_epi32(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shldi_epi32() {
        let a = _mm_set1_epi32(1);
        let b = _mm_set1_epi32(1 << 31);
        let r = _mm_mask_shldi_epi32::<2>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shldi_epi32::<2>(a, 0b00001111, a, b);
        let e = _mm_set1_epi32(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shldi_epi32() {
        let a = _mm_set1_epi32(1);
        let b = _mm_set1_epi32(1 << 31);
        let r = _mm_maskz_shldi_epi32::<2>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shldi_epi32::<2>(0b00001111, a, b);
        let e = _mm_set1_epi32(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shldi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1 << 15);
        let r = _mm512_shldi_epi16::<2>(a, b);
        let e = _mm512_set1_epi16(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shldi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1 << 15);
        let r = _mm512_mask_shldi_epi16::<2>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shldi_epi16::<2>(a, 0b11111111_11111111_11111111_11111111, a, b);
        let e = _mm512_set1_epi16(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shldi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1 << 15);
        let r = _mm512_maskz_shldi_epi16::<2>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shldi_epi16::<2>(0b11111111_11111111_11111111_11111111, a, b);
        let e = _mm512_set1_epi16(6);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shldi_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1 << 15);
        let r = _mm256_shldi_epi16::<2>(a, b);
        let e = _mm256_set1_epi16(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shldi_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1 << 15);
        let r = _mm256_mask_shldi_epi16::<2>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shldi_epi16::<2>(a, 0b11111111_11111111, a, b);
        let e = _mm256_set1_epi16(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shldi_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1 << 15);
        let r = _mm256_maskz_shldi_epi16::<2>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shldi_epi16::<2>(0b11111111_11111111, a, b);
        let e = _mm256_set1_epi16(6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shldi_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1 << 15);
        let r = _mm_shldi_epi16::<2>(a, b);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shldi_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1 << 15);
        let r = _mm_mask_shldi_epi16::<2>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shldi_epi16::<2>(a, 0b11111111, a, b);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shldi_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1 << 15);
        let r = _mm_maskz_shldi_epi16::<2>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shldi_epi16::<2>(0b11111111, a, b);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shrdi_epi64() {
        let a = _mm512_set1_epi64(2);
        let b = _mm512_set1_epi64(8);
        let r = _mm512_shrdi_epi64::<1>(a, b);
        let e = _mm512_set1_epi64(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shrdi_epi64() {
        let a = _mm512_set1_epi64(2);
        let b = _mm512_set1_epi64(8);
        let r = _mm512_mask_shrdi_epi64::<1>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shrdi_epi64::<1>(a, 0b11111111, a, b);
        let e = _mm512_set1_epi64(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shrdi_epi64() {
        let a = _mm512_set1_epi64(2);
        let b = _mm512_set1_epi64(8);
        let r = _mm512_maskz_shrdi_epi64::<1>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shrdi_epi64::<1>(0b11111111, a, b);
        let e = _mm512_set1_epi64(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shrdi_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(8);
        let r = _mm256_shrdi_epi64::<1>(a, b);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shrdi_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(8);
        let r = _mm256_mask_shrdi_epi64::<1>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shrdi_epi64::<1>(a, 0b00001111, a, b);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shrdi_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(8);
        let r = _mm256_maskz_shrdi_epi64::<1>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shrdi_epi64::<1>(0b00001111, a, b);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shrdi_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(8);
        let r = _mm_shrdi_epi64::<1>(a, b);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shrdi_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(8);
        let r = _mm_mask_shrdi_epi64::<1>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shrdi_epi64::<1>(a, 0b00000011, a, b);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shrdi_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(8);
        let r = _mm_maskz_shrdi_epi64::<1>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shrdi_epi64::<1>(0b00000011, a, b);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shrdi_epi32() {
        let a = _mm512_set1_epi32(2);
        let b = _mm512_set1_epi32(8);
        let r = _mm512_shrdi_epi32::<1>(a, b);
        let e = _mm512_set1_epi32(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shrdi_epi32() {
        let a = _mm512_set1_epi32(2);
        let b = _mm512_set1_epi32(8);
        let r = _mm512_mask_shrdi_epi32::<1>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shrdi_epi32::<1>(a, 0b11111111_11111111, a, b);
        let e = _mm512_set1_epi32(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shrdi_epi32() {
        let a = _mm512_set1_epi32(2);
        let b = _mm512_set1_epi32(8);
        let r = _mm512_maskz_shrdi_epi32::<1>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shrdi_epi32::<1>(0b11111111_11111111, a, b);
        let e = _mm512_set1_epi32(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shrdi_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(8);
        let r = _mm256_shrdi_epi32::<1>(a, b);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shrdi_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(8);
        let r = _mm256_mask_shrdi_epi32::<1>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shrdi_epi32::<1>(a, 0b11111111, a, b);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shrdi_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(8);
        let r = _mm256_maskz_shrdi_epi32::<1>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shrdi_epi32::<1>(0b11111111, a, b);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shrdi_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(8);
        let r = _mm_shrdi_epi32::<1>(a, b);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shrdi_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(8);
        let r = _mm_mask_shrdi_epi32::<1>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shrdi_epi32::<1>(a, 0b00001111, a, b);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shrdi_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(8);
        let r = _mm_maskz_shrdi_epi32::<1>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shrdi_epi32::<1>(0b00001111, a, b);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_shrdi_epi16() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(8);
        let r = _mm512_shrdi_epi16::<1>(a, b);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_shrdi_epi16() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(8);
        let r = _mm512_mask_shrdi_epi16::<1>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shrdi_epi16::<1>(a, 0b11111111_11111111_11111111_11111111, a, b);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_shrdi_epi16() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(8);
        let r = _mm512_maskz_shrdi_epi16::<1>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shrdi_epi16::<1>(0b11111111_11111111_11111111_11111111, a, b);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_shrdi_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(8);
        let r = _mm256_shrdi_epi16::<1>(a, b);
        let e = _mm256_set1_epi16(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_shrdi_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(8);
        let r = _mm256_mask_shrdi_epi16::<1>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shrdi_epi16::<1>(a, 0b11111111_11111111, a, b);
        let e = _mm256_set1_epi16(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_shrdi_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(8);
        let r = _mm256_maskz_shrdi_epi16::<1>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shrdi_epi16::<1>(0b11111111_11111111, a, b);
        let e = _mm256_set1_epi16(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_shrdi_epi16() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(8);
        let r = _mm_shrdi_epi16::<1>(a, b);
        let e = _mm_set1_epi16(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_shrdi_epi16() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(8);
        let r = _mm_mask_shrdi_epi16::<1>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shrdi_epi16::<1>(a, 0b11111111, a, b);
        let e = _mm_set1_epi16(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_shrdi_epi16() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(8);
        let r = _mm_maskz_shrdi_epi16::<1>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shrdi_epi16::<1>(0b11111111, a, b);
        let e = _mm_set1_epi16(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_expandloadu_epi16() {
        let src = _mm512_set1_epi16(42);
        let a = &[
            1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b11101000_11001010_11110000_00001111;
        let r = _mm512_mask_expandloadu_epi16(src, m, black_box(p));
        let e = _mm512_set_epi16(
            16, 15, 14, 42, 13, 42, 42, 42, 12, 11, 42, 42, 10, 42, 9, 42, 8, 7, 6, 5, 42, 42, 42,
            42, 42, 42, 42, 42, 4, 3, 2, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_expandloadu_epi16() {
        let a = &[
            1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b11101000_11001010_11110000_00001111;
        let r = _mm512_maskz_expandloadu_epi16(m, black_box(p));
        let e = _mm512_set_epi16(
            16, 15, 14, 0, 13, 0, 0, 0, 12, 11, 0, 0, 10, 0, 9, 0, 8, 7, 6, 5, 0, 0, 0, 0, 0, 0, 0,
            0, 4, 3, 2, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_expandloadu_epi16() {
        let src = _mm256_set1_epi16(42);
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm256_mask_expandloadu_epi16(src, m, black_box(p));
        let e = _mm256_set_epi16(8, 7, 6, 42, 5, 42, 42, 42, 4, 3, 42, 42, 2, 42, 1, 42);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_expandloadu_epi16() {
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm256_maskz_expandloadu_epi16(m, black_box(p));
        let e = _mm256_set_epi16(8, 7, 6, 0, 5, 0, 0, 0, 4, 3, 0, 0, 2, 0, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_expandloadu_epi16() {
        let src = _mm_set1_epi16(42);
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
        let p = a.as_ptr();
        let m = 0b11101000;
        let r = _mm_mask_expandloadu_epi16(src, m, black_box(p));
        let e = _mm_set_epi16(4, 3, 2, 42, 1, 42, 42, 42);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_expandloadu_epi16() {
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
        let p = a.as_ptr();
        let m = 0b11101000;
        let r = _mm_maskz_expandloadu_epi16(m, black_box(p));
        let e = _mm_set_epi16(4, 3, 2, 0, 1, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_expandloadu_epi8() {
        let src = _mm512_set1_epi8(42);
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        let p = a.as_ptr();
        let m = 0b11101000_11001010_11110000_00001111_11111111_00000000_10101010_01010101;
        let r = _mm512_mask_expandloadu_epi8(src, m, black_box(p));
        let e = _mm512_set_epi8(
            32, 31, 30, 42, 29, 42, 42, 42, 28, 27, 42, 42, 26, 42, 25, 42, 24, 23, 22, 21, 42, 42,
            42, 42, 42, 42, 42, 42, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 42, 42, 42, 42,
            42, 42, 42, 42, 8, 42, 7, 42, 6, 42, 5, 42, 42, 4, 42, 3, 42, 2, 42, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_expandloadu_epi8() {
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        let p = a.as_ptr();
        let m = 0b11101000_11001010_11110000_00001111_11111111_00000000_10101010_01010101;
        let r = _mm512_maskz_expandloadu_epi8(m, black_box(p));
        let e = _mm512_set_epi8(
            32, 31, 30, 0, 29, 0, 0, 0, 28, 27, 0, 0, 26, 0, 25, 0, 24, 23, 22, 21, 0, 0, 0, 0, 0,
            0, 0, 0, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0,
            7, 0, 6, 0, 5, 0, 0, 4, 0, 3, 0, 2, 0, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_expandloadu_epi8() {
        let src = _mm256_set1_epi8(42);
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b11101000_11001010_11110000_00001111;
        let r = _mm256_mask_expandloadu_epi8(src, m, black_box(p));
        let e = _mm256_set_epi8(
            16, 15, 14, 42, 13, 42, 42, 42, 12, 11, 42, 42, 10, 42, 9, 42, 8, 7, 6, 5, 42, 42, 42,
            42, 42, 42, 42, 42, 4, 3, 2, 1,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_expandloadu_epi8() {
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b11101000_11001010_11110000_00001111;
        let r = _mm256_maskz_expandloadu_epi8(m, black_box(p));
        let e = _mm256_set_epi8(
            16, 15, 14, 0, 13, 0, 0, 0, 12, 11, 0, 0, 10, 0, 9, 0, 8, 7, 6, 5, 0, 0, 0, 0, 0, 0, 0,
            0, 4, 3, 2, 1,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_expandloadu_epi8() {
        let src = _mm_set1_epi8(42);
        let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm_mask_expandloadu_epi8(src, m, black_box(p));
        let e = _mm_set_epi8(8, 7, 6, 42, 5, 42, 42, 42, 4, 3, 42, 42, 2, 42, 1, 42);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_expandloadu_epi8() {
        let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm_maskz_expandloadu_epi8(m, black_box(p));
        let e = _mm_set_epi8(8, 7, 6, 0, 5, 0, 0, 0, 4, 3, 0, 0, 2, 0, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_compressstoreu_epi16() {
        let a = _mm512_set_epi16(
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
            10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        );
        let mut r = [0_i16; 32];
        _mm512_mask_compressstoreu_epi16(r.as_mut_ptr(), 0, a);
        assert_eq!(&r, &[0_i16; 32]);
        _mm512_mask_compressstoreu_epi16(r.as_mut_ptr(), 0b11110000_11001010_11111111_00000000, a);
        assert_eq!(
            &r,
            &[
                9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 24, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_compressstoreu_epi16() {
        let a = _mm256_set_epi16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let mut r = [0_i16; 16];
        _mm256_mask_compressstoreu_epi16(r.as_mut_ptr(), 0, a);
        assert_eq!(&r, &[0_i16; 16]);
        _mm256_mask_compressstoreu_epi16(r.as_mut_ptr(), 0b11110000_11001010, a);
        assert_eq!(&r, &[2, 4, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_compressstoreu_epi16() {
        let a = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        let mut r = [0_i16; 8];
        _mm_mask_compressstoreu_epi16(r.as_mut_ptr(), 0, a);
        assert_eq!(&r, &[0_i16; 8]);
        _mm_mask_compressstoreu_epi16(r.as_mut_ptr(), 0b11110000, a);
        assert_eq!(&r, &[5, 6, 7, 8, 0, 0, 0, 0]);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_compressstoreu_epi8() {
        let a = _mm512_set_epi8(
            64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
            42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
            20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        );
        let mut r = [0_i8; 64];
        _mm512_mask_compressstoreu_epi8(r.as_mut_ptr(), 0, a);
        assert_eq!(&r, &[0_i8; 64]);
        _mm512_mask_compressstoreu_epi8(
            r.as_mut_ptr(),
            0b11110000_11001010_11111111_00000000_10101010_01010101_11110000_00001111,
            a,
        );
        assert_eq!(
            &r,
            &[
                1, 2, 3, 4, 13, 14, 15, 16, 17, 19, 21, 23, 26, 28, 30, 32, 41, 42, 43, 44, 45, 46,
                47, 48, 50, 52, 55, 56, 61, 62, 63, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_compressstoreu_epi8() {
        let a = _mm256_set_epi8(
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
            10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        );
        let mut r = [0_i8; 32];
        _mm256_mask_compressstoreu_epi8(r.as_mut_ptr(), 0, a);
        assert_eq!(&r, &[0_i8; 32]);
        _mm256_mask_compressstoreu_epi8(r.as_mut_ptr(), 0b11110000_11001010_11111111_00000000, a);
        assert_eq!(
            &r,
            &[
                9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 24, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_compressstoreu_epi8() {
        let a = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let mut r = [0_i8; 16];
        _mm_mask_compressstoreu_epi8(r.as_mut_ptr(), 0, a);
        assert_eq!(&r, &[0_i8; 16]);
        _mm_mask_compressstoreu_epi8(r.as_mut_ptr(), 0b11110000_11001010, a);
        assert_eq!(&r, &[2, 4, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}

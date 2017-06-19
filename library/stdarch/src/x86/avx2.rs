use v256::*;
use v128::*;
use x86::__m256i;

/// Computes the absolute values of packed 32-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi32(a: i32x8) -> i32x8 {
    unsafe { pabsd(a) }
}

/// Computes the absolute values of packed 16-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi16(a: i16x16) -> i16x16 {
    unsafe { pabsw(a) }
}

/// Computes the absolute values of packed 8-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi8(a: i8x32) -> i8x32 {
    unsafe { pabsb(a) }
}

/// Add packed 64-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a + b
}

/// Add packed 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a + b
}

/// Add packed 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a + b
}

/// Add packed 8-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a + b
}

/// Add packed 8-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epi8(a: i8x32, b: i8x32) -> i8x32 {
    unsafe { paddsb(a, b) }
}

/// Add packed 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { paddsw(a, b) }
}

/// Add packed unsigned 8-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epu8(a: u8x32, b: u8x32) -> u8x32 {
    unsafe { paddusb(a, b) }
}

/// Add packed unsigned 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epu16(a: u16x16, b: u16x16) -> u16x16 {
    unsafe { paddusw(a, b) }
}

// TODO _mm256_alignr_epi8

/// Compute the bitwise AND of 256 bits (representing integer data)
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
    a & b
}

/// Compute the bitwise NOT of 256 bits (representing integer data)
/// in `a` and then AND with `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_andnot_si256(a: __m256i, b: __m256i) -> __m256i {
    (!a) & b
}

/// Average packed unsigned 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_avg_epu16 (a: u16x16, b: u16x16) -> u16x16 {
    unsafe { pavgw(a, b) }
}

/// Average packed unsigned 8-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_avg_epu8 (a: u8x32, b: u8x32) -> u8x32 {
    unsafe { pavgb(a, b) }
}

// TODO _mm256_blend_epi16
// TODO _mm_blend_epi32
// TODO _mm256_blend_epi32

/// Blend packed 8-bit integers from `a` and `b` using `mask`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_blendv_epi8(a:i8x32,b:i8x32,mask:__m256i) -> i8x32 {
    unsafe { pblendvb(a,b,mask) }
}

// TODO _mm_broadcastb_epi8
// TODO _mm256_broadcastb_epi8
// TODO _mm_broadcastd_epi32
// TODO _mm256_broadcastd_epi32
// TODO _mm_broadcastq_epi64
// TODO _mm256_broadcastq_epi64
// TODO _mm_broadcastsd_pd
// TODO _mm256_broadcastsd_pd
// TODO _mm_broadcastsi128_si256
// TODO _mm256_broadcastsi128_si256
// TODO _mm_broadcastss_ps
// TODO _mm256_broadcastss_ps
// TODO _mm_broadcastw_epi16
// TODO _mm256_broadcastw_epi16
// TODO _mm256_bslli_epi128
// TODO _mm256_bsrli_epi128


/// Compare packed 64-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a.eq(b)
}

/// Compare packed 32-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a.eq(b)
}

/// Compare packed 16-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a.eq(b)
}

/// Compare packed 8-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a.eq(b)
}

/// Compare packed 64-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a.gt(b)
}

/// Compare packed 32-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a.gt(b)
}

/// Compare packed 16-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a.gt(b)
}

/// Compare packed 8-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a.gt(b)
}

// TODO _mm256_cvtepi16_epi32
// TODO _mm256_cvtepi16_epi64
// TODO _mm256_cvtepi32_epi64
// TODO _mm256_cvtepi8_epi16
// TODO _mm256_cvtepi8_epi32
// TODO _mm256_cvtepi8_epi64
// TODO _mm256_cvtepu16_epi32
// TODO _mm256_cvtepu16_epi64
// TODO _mm256_cvtepu32_epi64
// TODO _mm256_cvtepu8_epi16
// TODO _mm256_cvtepu8_epi32
// TODO _mm256_cvtepu8_epi64
// TODO _m128i _mm256_extracti128_si256

/// Horizontally add adjacent pairs of 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hadd_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phaddw(a, b) }
}

/// Horizontally add adjacent pairs of 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hadd_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { phaddd(a, b) }
}

/// Horizontally add adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hadds_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phaddsw(a, b) }
}

/// Horizontally substract adjacent pairs of 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hsub_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phsubw(a, b) }
}

/// Horizontally substract adjacent pairs of 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hsub_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { phsubd(a, b) }
}

/// Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hsubs_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phsubsw(a, b) }
}


// TODO _mm_i32gather_epi32 (int const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i32gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale)
// TODO _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
// TODO _mm256_mask_i32gather_epi32 (__m256i src, int const* base_addr, __m256i vindex, __m256i mask, const int scale)
// TODO _mm_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i32gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale)
// TODO _mm256_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// TODO _mm256_mask_i32gather_epi64 (__m256i src, __int64 const* base_addr, __m128i vindex, __m256i mask, const int scale)
// TODO _mm_i32gather_pd (double const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i32gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
// TODO _mm256_i32gather_pd (double const* base_addr, __m128i vindex, const int scale)
// TODO _mm256_mask_i32gather_pd (__m256d src, double const* base_addr, __m128i vindex, __m256d mask, const int scale)
// TODO _mm_i32gather_ps (float const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i32gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale)
// TODO _mm256_i32gather_ps (float const* base_addr, __m256i vindex, const int scale)
// TODO _mm256_mask_i32gather_ps (__m256 src, float const* base_addr, __m256i vindex, __m256 mask, const int scale)
// TODO _mm_i64gather_epi32 (int const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale)
// TODO _mm256_i64gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
// TODO _mm256_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m256i vindex, __m128i mask, const int scale)
// TODO _mm_i64gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i64gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale)
// TODO _mm256_i64gather_epi64 (__int64 const* base_addr, __m256i vindex, const int scale)
// TODO _mm256_mask_i64gather_epi64 (__m256i src, __int64 const* base_addr, __m256i vindex, __m256i mask, const int scale)
// TODO _mm_i64gather_pd (double const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i64gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
// TODO _mm256_i64gather_pd (double const* base_addr, __m256i vindex, const int scale)
// TODO _mm256_mask_i64gather_pd (__m256d src, double const* base_addr, __m256i vindex, __m256d mask, const int scale)
// TODO _mm_i64gather_ps (float const* base_addr, __m128i vindex, const int scale)
// TODO _mm_mask_i64gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale)
// TODO _mm256_i64gather_ps (float const* base_addr, __m256i vindex, const int scale)
// TODO _mm256_mask_i64gather_ps
// TODO _mm256_inserti128_si256

/// Multiply packed signed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Horizontally add adjacent pairs
/// of intermediate 32-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_madd_epi16(a: i16x16, b: i16x16) -> i32x8 {
    unsafe { pmaddwd(a, b) }
}

/// Vertically multiply each unsigned 8-bit integer from `a` with the
/// corresponding signed 8-bit integer from `b`, producing intermediate
/// signed 16-bit integers. Horizontally add adjacent pairs of intermediate
/// signed 16-bit integers
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_maddubs_epi16(a: u8x32, b: u8x32) -> i16x16 {
    unsafe { pmaddubsw(a, b) }
}

// TODO _mm_maskload_epi32 (int const* mem_addr, __m128i mask)
// TODO _mm256_maskload_epi32 (int const* mem_addr, __m256i mask)
// TODO _mm_maskload_epi64 (__int64 const* mem_addr, __m128i mask)
// TODO _mm256_maskload_epi64 (__int64 const* mem_addr, __m256i mask)
// TODO _mm_maskstore_epi32 (int* mem_addr, __m128i mask, __m128i a)
// TODO _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a)
// TODO _mm_maskstore_epi64 (__int64* mem_addr, __m128i mask, __m128i a)
// TODO _mm256_maskstore_epi64 (__int64* mem_addr, __m256i mask, __m256i a)

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_max_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { pmaxsw(a, b) }
}

/// Compare packed 32-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_max_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { pmaxsd(a, b) }
}

/// Compare packed 8-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_max_epi8(a: i8x32, b: i8x32) -> i8x32 {
    unsafe { pmaxsb(a, b) }
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return
/// the packed maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_max_epu16(a: u16x16, b: u16x16) -> u16x16 {
    unsafe { pmaxuw(a, b) }
}

/// Compare packed unsigned 32-bit integers in `a` and `b`, and return
/// the packed maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_max_epu32(a: u32x8, b: u32x8) -> u32x8 {
    unsafe { pmaxud(a, b) }
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return
/// the packed maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_max_epu8(a: u8x32, b: u8x32) -> u8x32 {
    unsafe { pmaxub(a, b) }
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_min_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { pminsw(a, b) }
}

/// Compare packed 32-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_min_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { pminsd(a, b) }
}

/// Compare packed 8-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_min_epi8(a: i8x32, b: i8x32) -> i8x32 {
    unsafe { pminsb(a, b) }
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return
/// the packed minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_min_epu16(a: u16x16, b: u16x16) -> u16x16 {
    unsafe { pminuw(a, b) }
}

/// Compare packed unsigned 32-bit integers in `a` and `b`, and return
/// the packed minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_min_epu32(a: u32x8, b: u32x8) -> u32x8 {
    unsafe { pminud(a, b) }
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return
/// the packed minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_min_epu8(a: u8x32, b: u8x32) -> u8x32 {
    unsafe { pminub(a, b) }
}

/*** The following two functions fail in debug, but work in release

/// Create mask from the most significant bit of each 8-bit element in `a`,
/// return the result.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_movemask_epi8(a: i8x32) -> i32 {
    unsafe { pmovmskb(a) }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned
/// 8-bit integers in `a` compared to those in `b`, and store the 16-bit
/// results in dst. Eight SADs are performed for each 128-bit lane using one
/// quadruplet from `b` and eight quadruplets from `a`. One quadruplet is
/// selected from `b` starting at on the offset specified in `imm8`. Eight
/// quadruplets are formed from sequential 8-bit integers selected from `a`
/// starting at the offset specified in `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mpsadbw_epu8(a: u8x32, b: u8x32, imm8: i32) -> u16x16 {
    unsafe { mpsadbw(a, b, imm8) }
}

***/

/// Multiply the low 32-bit integers from each packed 64-bit element in
/// `a` and `b`
///
/// Return the 64-bit results.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mul_epi32(a: i32x8, b: i32x8) -> i64x4 {
    unsafe { pmuldq(a, b) }
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit
/// element in `a` and `b`
///
/// Return the unsigned 64-bit results.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mul_epu32(a: u32x8, b: u32x8) -> u64x4 {
    unsafe { pmuludq(a, b) }
}

/// Multiply the packed 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers and returning the high 16 bits of the
/// intermediate integers.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mulhi_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { pmulhw(a, b) }
}

/// Multiply the packed unsigned 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers and returning the high 16 bits of the
/// intermediate integers.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mulhi_epu16(a: u16x16, b: u16x16) -> u16x16 {
    unsafe { pmulhuw(a, b) }
}

/// Multiply the packed 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers, and return the low 16 bits of the
/// intermediate integers
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mullo_epi16(a: i16x16, b:i16x16) -> i16x16 {
    a * b
}


/// Multiply the packed 32-bit integers in `a` and `b`, producing
/// intermediate 64-bit integers, and return the low 16 bits of the
/// intermediate integers
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mullo_epi32(a: i32x8, b:i32x8) -> i32x8 {
    a * b
}

/// Multiply packed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Truncate each intermediate
/// integer to the 18 most significant bits, round by adding 1, and
/// return bits [16:1]
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_mulhrs_epi16(a: i16x16, b:i16x16) -> i16x16 {
    unsafe { pmulhrsw(a, b) }
}

/// Compute the bitwise OR of 256 bits (representing integer data) in `a`
/// and `b`
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_or_si256(a: __m256i, b: __m256i) -> __m256i {
    a | b
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using signed saturation
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_packs_epi16(a: i16x16, b: i16x16) -> i8x32 {
    unsafe { packsswb(a, b) }
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using signed saturation
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_packs_epi32(a: i32x8, b: i32x8) -> i16x16 {
    unsafe { packssdw(a, b) }
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using unsigned saturation
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_packus_epi16(a: i16x16, b: i16x16) -> u8x32 {
    unsafe { packuswb(a, b) }
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using unsigned saturation
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_packus_epi32(a: i32x8, b: i32x8) -> u16x16 {
    unsafe { packusdw(a, b) }
}

// TODO _mm256_permute2x128_si256 (__m256i a, __m256i b, const int imm8)
// TODO _mm256_permute4x64_epi64 (__m256i a, const int imm8)
// TODO _mm256_permute4x64_pd (__m256d a, const int imm8)
// TODO _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
// TODO _mm256_permutevar8x32_ps (__m256 a, __m256i idx)

/// Compute the absolute differences of packed unsigned 8-bit integers in `a`
/// and `b`, then horizontally sum each consecutive 8 differences to
/// produce four unsigned 16-bit integers, and pack these unsigned 16-bit
/// integers in the low 16 bits of the 64-bit return value
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sad_epu8 (a: u8x32, b: u8x32) -> u64x4 {
    unsafe { psadbw(a, b) }
}

// TODO _mm256_shuffle_epi32 (__m256i a, const int imm8)
// TODO _mm256_shuffle_epi8 (__m256i a, __m256i b)
// TODO _mm256_shufflehi_epi16 (__m256i a, const int imm8)
// TODO _mm256_shufflelo_epi16 (__m256i a, const int imm8)

/// Negate packed 16-bit integers in `a` when the corresponding signed
/// 16-bit integer in `b` is negative, and return the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sign_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { psignw(a, b) }
}

/// Negate packed 32-bit integers in `a` when the corresponding signed
/// 32-bit integer in `b` is negative, and return the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sign_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { psignd(a, b) }
}

/// Negate packed 8-bit integers in `a` when the corresponding signed
/// 8-bit integer in `b` is negative, and return the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sign_epi8(a: i8x32, b: i8x32) -> i8x32 {
    unsafe { psignb(a, b) }
}

/// Shift packed 16-bit integers in `a` left by `count` while
/// shifting in zeros, and return the result
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sll_epi16(a: i16x16, count: i16x8) -> i16x16 {
    unsafe { psllw(a, count) }
}

/// Shift packed 32-bit integers in `a` left by `count` while
/// shifting in zeros, and return the result
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sll_epi32(a: i32x8, count: i32x4) -> i32x8 {
    unsafe { pslld(a, count) }
}

/// Shift packed 64-bit integers in `a` left by `count` while
/// shifting in zeros, and return the result
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sll_epi64(a: i64x4, count: i64x2) -> i64x4 {
    unsafe { psllq(a, count) }
}

/// Shift packed 16-bit integers in `a` left by `imm8` while
/// shifting in zeros, return the results;
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_slli_epi16(a: i16x16, imm8: i32) -> i16x16 {
    unsafe { pslliw(a, imm8) }
}

/// Shift packed 32-bit integers in `a` left by `imm8` while
/// shifting in zeros, return the results;
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_slli_epi32(a: i32x8, imm8: i32) -> i32x8 {
    unsafe { psllid(a, imm8) }
}

/// Shift packed 64-bit integers in `a` left by `imm8` while
/// shifting in zeros, return the results;
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_slli_epi64(a: i64x4, imm8: i32) -> i64x4 {
    unsafe { pslliq(a, imm8) }
}

// TODO _mm256_slli_si256 (__m256i a, const int imm8)

/// Shift packed 32-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm_sllv_epi32(a: i32x4, count: i32x4) -> i32x4 {
    unsafe { psllvd(a, count) }
}

/// Shift packed 32-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sllv_epi32(a: i32x8, count: i32x8) -> i32x8 {
    unsafe { psllvd256(a, count) }
}

/// Shift packed 64-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm_sllv_epi64(a: i64x2, count: i64x2) -> i64x2 {
    unsafe { psllvq(a, count) }
}

/// Shift packed 64-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sllv_epi64(a: i64x4, count: i64x4) -> i64x4 {
    unsafe { psllvq256(a, count) }
}

/// Shift packed 16-bit integers in `a` right by `count` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sra_epi16(a: i16x16, count: i16x8) -> i16x16 {
    unsafe { psraw(a, count) }
}

/// Shift packed 32-bit integers in `a` right by `count` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sra_epi32(a: i32x8, count: i32x4) -> i32x8 {
    unsafe { psrad(a, count) }
}

/// Shift packed 16-bit integers in `a` right by `imm8` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srai_epi16(a: i16x16, imm8: i32) -> i16x16 {
    unsafe { psraiw(a, imm8) }
}

/// Shift packed 32-bit integers in `a` right by `imm8` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srai_epi32(a: i32x8, imm8: i32) -> i32x8 {
    unsafe { psraid(a, imm8) }
}

/// Shift packed 32-bit integers in `a` right by the amount specified by the
/// corresponding element in `count` while shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm_srav_epi32(a: i32x4, count: i32x4) -> i32x4 {
    unsafe { psravd(a, count) }
}

/// Shift packed 32-bit integers in `a` right by the amount specified by the
/// corresponding element in `count` while shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srav_epi32(a: i32x8, count: i32x8) -> i32x8 {
    unsafe { psravd256(a, count) }
}


/// Shift packed 16-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srl_epi16(a: i16x16, count: i16x8) -> i16x16 {
    unsafe { psrlw(a, count) }
}

/// Shift packed 32-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srl_epi32(a: i32x8, count: i32x4) -> i32x8 {
    unsafe { psrld(a, count) }
}

/// Shift packed 64-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srl_epi64(a: i64x4, count: i64x2) -> i64x4 {
    unsafe { psrlq(a, count) }
}

/// Shift packed 16-bit integers in `a` right by `imm8` while shifting in
/// zeros
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srli_epi16(a: i16x16, imm8: i32) -> i16x16 {
    unsafe { psrliw(a, imm8) }
}

/// Shift packed 32-bit integers in `a` right by `imm8` while shifting in
/// zeros
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srli_epi32(a: i32x8, imm8: i32) -> i32x8 {
    unsafe { psrlid(a, imm8) }
}

/// Shift packed 64-bit integers in `a` right by `imm8` while shifting in
/// zeros
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srli_epi64(a: i64x4, imm8: i32) -> i64x4 {
    unsafe { psrliq(a, imm8) }
}

/// Shift packed 32-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm_srlv_epi32(a: i32x4, count: i32x4) -> i32x4 {
    unsafe { psrlvd(a, count) }
}

/// Shift packed 32-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srlv_epi32(a: i32x8, count: i32x8) -> i32x8 {
    unsafe { psrlvd256(a, count) }
}

/// Shift packed 64-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm_srlv_epi64(a: i64x2, count: i64x2) -> i64x2 {
    unsafe { psrlvq(a, count) }
}

/// Shift packed 64-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_srlv_epi64(a: i64x4, count: i64x4) -> i64x4 {
    unsafe { psrlvq256(a, count) }
}

// TODO _mm256_stream_load_si256 (__m256i const* mem_addr)

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sub_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a - b
}

/// Subtract packed 32-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sub_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a - b
}

/// Subtract packed 64-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sub_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a - b
}

/// Subtract packed 8-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_sub_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a - b
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in
/// `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_subs_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { psubsw(a, b) }
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in
/// `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_subs_epi8(a: i8x32, b: i8x32) -> i8x32 {
    unsafe { psubsb(a, b) }
}

/// Subtract packed unsigned 16-bit integers in `b` from packed 16-bit
/// integers in `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_subs_epu16(a: u16x16, b: u16x16) -> u16x16 {
    unsafe { psubusw(a, b) }
}

/// Subtract packed unsigned 8-bit integers in `b` from packed 8-bit
/// integers in `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_subs_epu8(a: u8x32, b: u8x32) -> u8x32 {
    unsafe { psubusb(a, b) }
}

// TODO __mm256_unpackhi_epi16 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpackhi_epi32 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpackhi_epi64 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpackhi_epi8 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpacklo_epi16 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpacklo_epi32 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpacklo_epi64 (__m256i a, __m256i b)
// TODO __m256i _mm256_unpacklo_epi8 (__m256i a, __m256i b)

/// Compute the bitwise XOR of 256 bits (representing integer data)
/// in `a` and `b`
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i {
    a ^ b
}


#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx2.pabs.b"]
    fn pabsb(a: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pabs.w"]
    fn pabsw(a: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pabs.d"]
    fn pabsd(a: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.padds.b"]
    fn paddsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.padds.w"]
    fn paddsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.paddus.b"]
    fn paddusb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.paddus.w"]
    fn paddusw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pavg.b"]
    fn pavgb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pavg.w"]
    fn pavgw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pblendvb"]
    fn pblendvb(a: i8x32, b: i8x32, mask: __m256i) -> i8x32;
    #[link_name = "llvm.x86.avx2.phadd.w"]
    fn phaddw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.phadd.d"]
    fn phaddd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.phadd.sw"]
    fn phaddsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.phsub.w"]
    fn phsubw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.phsub.d"]
    fn phsubd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.phsub.sw"]
    fn phsubsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmadd.wd"]
    fn pmaddwd(a: i16x16, b: i16x16) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmadd.ub.sw"]
    fn pmaddubsw(a: u8x32, b: u8x32) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmaxs.w"]
    fn pmaxsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmaxs.d"]
    fn pmaxsd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmaxs.b"]
    fn pmaxsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pmaxu.w"]
    fn pmaxuw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pmaxu.d"]
    fn pmaxud(a: u32x8, b: u32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.pmaxu.b"]
    fn pmaxub(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pmins.w"]
    fn pminsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmins.d"]
    fn pminsd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmins.b"]
    fn pminsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pminu.w"]
    fn pminuw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pminu.d"]
    fn pminud(a: u32x8, b: u32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.pminu.b"]
    fn pminub(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pmovmskb"]  //fails in debug
    fn pmovmskb(a: i8x32) -> i32;
    #[link_name = "llvm.x86.avx2.mpsadbw"] //fails in debug
    fn mpsadbw(a: u8x32, b: u8x32, imm8: i32) -> u16x16;
    #[link_name = "llvm.x86.avx2.pmulhu.w"]
    fn pmulhuw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pmulh.w"]
    fn pmulhw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmul.dq"]
    fn pmuldq(a: i32x8, b:i32x8) -> i64x4;
    #[link_name = "llvm.x86.avx2.pmulu.dq"]
    fn pmuludq(a: u32x8, b:u32x8) -> u64x4;
    #[link_name = "llvm.x86.avx2.pmul.hr.sw"]
    fn pmulhrsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.packsswb"]
    fn packsswb(a: i16x16, b: i16x16) -> i8x32;
    #[link_name = "llvm.x86.avx2.packssdw"]
    fn packssdw(a: i32x8, b: i32x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.packuswb"]
    fn packuswb(a: i16x16, b: i16x16) -> u8x32;
    #[link_name = "llvm.x86.avx2.packusdw"]
    fn packusdw(a: i32x8, b: i32x8) -> u16x16;
    #[link_name = "llvm.x86.avx2.psad.bw"]
    fn psadbw(a: u8x32, b: u8x32) -> u64x4;
    #[link_name = "llvm.x86.avx2.psign.b"]
    fn psignb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.psign.w"]
    fn psignw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.psign.d"]
    fn psignd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psll.w"]
    fn psllw(a: i16x16, count: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psll.d"]
    fn pslld(a: i32x8, count: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psll.q"]
    fn psllq(a: i64x4, count: i64x2) -> i64x4;
    #[link_name = "llvm.x86.avx2.pslli.w"]
    fn pslliw(a: i16x16, imm8: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.pslli.d"]
    fn psllid(a: i32x8, imm8: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.pslli.q"]
    fn pslliq(a: i64x4, imm8: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psllv.d"]
    fn psllvd(a:i32x4, count:i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psllv.d.256"]
    fn psllvd256(a:i32x8, count:i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psllv.q"]
    fn psllvq(a:i64x2, count:i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.psllv.q.256"]
    fn psllvq256(a:i64x4, count:i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.psra.w"]
    fn psraw(a: i16x16, count:i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psra.d"]
    fn psrad(a: i32x8, count:i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrai.w"]
    fn psraiw(a: i16x16, imm8: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrai.d"]
    fn psraid(a: i32x8, imm8: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrav.d"]
    fn psravd(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psrav.d.256"]
    fn psravd256(a: i32x8, count: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrl.w"]
    fn psrlw(a: i16x16, count:i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrl.d"]
    fn psrld(a: i32x8, count:i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrl.q"]
    fn psrlq(a: i64x4, count:i64x2) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrli.w"]
    fn psrliw(a: i16x16, imm8: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrli.d"]
    fn psrlid(a: i32x8, imm8: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrli.q"]
    fn psrliq(a: i64x4, imm8: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrlv.d"]
    fn psrlvd(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psrlv.d.256"]
    fn psrlvd256(a: i32x8, count: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrlv.q"]
    fn psrlvq(a: i64x2, count: i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.psrlv.q.256"]
    fn psrlvq256(a: i64x4, count: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.psubs.b"]
    fn psubsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.psubs.w"]
    fn psubsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.psubus.b"]
    fn psubusb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.psubus.w"]
    fn psubusw(a: u16x16, b: u16x16) -> u16x16;

}


#[cfg(test)]
mod tests {
    use v256::*;
    use v128::*;
    use x86::avx2;
    use x86::__m256i;
    use std;

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_abs_epi32() {
        let a = i32x8::new(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi32(a);
        let e = i32x8::new(
            0, 1, 1, std::i32::MAX,
            (std::i32::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_abs_epi16() {
        let a = i16x16::new(
            0, 1, -1, 2,
            -2, 3, -3, 4,
            -4, 5, -5, std::i16::MAX,
            std::i16::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi16(a);
        let e = i16x16::new(
            0, 1, 1, 2,
            2, 3, 3, 4,
            4, 5, 5, std::i16::MAX,
            (std::i16::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_abs_epi8() {
        let a = i8x32::new(
            0, 1, -1, 2,
            -2, 3, -3, 4,
            -4, 5, -5, std::i8::MAX,
            std::i8::MIN + 1, 100, -100, -32,
            0, 1, -1, 2,
            -2, 3, -3, 4,
            -4, 5, -5, std::i8::MAX,
            std::i8::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi8(a);
        let e = i8x32::new(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i8::MAX, (std::i8::MIN + 1).abs(), 100, 100, 32,
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i8::MAX, (std::i8::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi64() {
        let a = i64x4::new(-10, 0, 100, 1_000_000_000);
        let b = i64x4::new(-1, 0, 1, 2);
        let r = avx2::_mm256_add_epi64(a, b);
        let e = i64x4::new(-11, 0, 101, 1_000_000_002);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi32() {
        let a = i32x8::new(-1, 0, 1, 2, 3, 4, 5, 6);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_add_epi32(a, b);
        let e = i32x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15);
        let r = avx2::_mm256_add_epi16(a, b);
        let e = i16x16::new(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi8() {
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31);
        let r = avx2::_mm256_add_epi8(a, b);
        let e = i8x32::new(
            0, 2, 4, 6, 8, 10, 12, 14, 16,
            18, 20, 22, 24, 26, 28, 30, 32,
            34, 36, 38, 40, 42, 44, 46, 48,
            50, 52, 54, 56, 58, 60, 62);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi8() {
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = avx2::_mm256_adds_epi8(a, b);
        let e = i8x32::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi8_saturate_positive() {
        let a = i8x32::splat(0x7F);
        let b = i8x32::splat(1);
        let r = avx2::_mm256_adds_epi8(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi8_saturate_negative() {
        let a = i8x32::splat(-0x80);
        let b = i8x32::splat(-1);
        let r = avx2::_mm256_adds_epi8(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47);
        let r = avx2::_mm256_adds_epi16(a,  b);
        let e = i16x16::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi16_saturate_positive() {
        let a = i16x16::splat(0x7FFF);
        let b = i16x16::splat(1);
        let r = avx2::_mm256_adds_epi16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi16_saturate_negative() {
        let a = i16x16::splat(-0x8000);
        let b = i16x16::splat(-1);
        let r = avx2::_mm256_adds_epi16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu8() {
        let a = u8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = u8x32::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = avx2::_mm256_adds_epu8(a, b);
        let e = u8x32::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu8_saturate() {
        let a = u8x32::splat(0xFF);
        let b = u8x32::splat(1);
        let r = avx2::_mm256_adds_epu8(a, b);
        assert_eq!(r, a);
    }


    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu16() {
        let a = u16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u16x16::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47);
        let r = avx2::_mm256_adds_epu16(a, b);
        let e = u16x16::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu16_saturate() {
        let a = u16x16::splat(0xFFFF);
        let b = u16x16::splat(1);
        let r = avx2::_mm256_adds_epu16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_and_si256() {
        assert_eq!(
            avx2::_mm256_and_si256(
                __m256i::splat(5), __m256i::splat(3)),__m256i::splat(1));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_andnot_si256() {
        assert_eq!(
            avx2::_mm256_andnot_si256(__m256i::splat(5), __m256i::splat(3)),
            __m256i::splat(2));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_avg_epu8() {
        let (a, b) = (u8x32::splat(3), u8x32::splat(9));
        let r = avx2::_mm256_avg_epu8(a, b);
        assert_eq!(r, u8x32::splat(6));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_avg_epu16() {
        let (a, b) = (u16x16::splat(3), u16x16::splat(9));
        let r = avx2::_mm256_avg_epu16(a, b);
        assert_eq!(r, u16x16::splat(6));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_blendv_epi8() {
        let (a,b) = (i8x32::splat(4),i8x32::splat(2));
        let mask = i8x32::splat(0).replace(2,-1);
        let e = i8x32::splat(4).replace(2,2);
        let r= avx2::_mm256_blendv_epi8(a,b,mask);
        assert_eq!(r,e);
    }

    #[test]
    fn _mm256_cmpeq_epi8() {
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(
            31, 30, 2, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi8(a, b);
        assert_eq!(r, i8x32::splat(0).replace(2,0xFFu8 as i8));
    }

    #[test]
    fn _mm256_cmpeq_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(
            15, 14, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi16(a, b);
        assert_eq!(r, i16x16::splat(0).replace(2, 0xFFFFu16 as i16));
    }

    #[test]
    fn _mm256_cmpeq_epi32() {
        let a = i32x8::new(0, 1, 2, 3,4,5,6,7);
        let b = i32x8::new(7,6,2,4,3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi32(a, b);
        assert_eq!(r, i32x8::splat(0).replace(2, 0xFFFFFFFFu32 as i32));
    }

    #[test]
    fn _mm256_cmpeq_epi64() {
        let a = i64x4::new(0, 1, 2, 3);
        let b = i64x4::new(3, 2, 2, 0);
        let r = avx2::_mm256_cmpeq_epi64(a, b);
        assert_eq!(r, i64x4::splat(0).replace(
            2, 0xFFFFFFFFFFFFFFFFu64 as i64));
    }

    #[test]
    fn _mm256_cmpgt_epi8() {
        let a = i8x32::splat(0).replace(0, 5);
        let b = i8x32::splat(0);
        let r = avx2::_mm256_cmpgt_epi8(a, b);
        assert_eq!(r, i8x32::splat(0).replace(0, 0xFFu8 as i8));
    }

    #[test]
    fn _mm256_cmpgt_epi16() {
        let a = i16x16::splat(0).replace(0, 5);
        let b = i16x16::splat(0);
        let r = avx2::_mm256_cmpgt_epi16(a, b);
        assert_eq!(r, i16x16::splat(0).replace(0, 0xFFFFu16 as i16));
    }

    #[test]
    fn _mm256_cmpgt_epi32() {
        let a = i32x8::splat(0).replace(0, 5);
        let b = i32x8::splat(0);
        let r = avx2::_mm256_cmpgt_epi32(a, b);
        assert_eq!(r, i32x8::splat(0).replace(0, 0xFFFFFFFFu32 as i32));
    }

    #[test]
    fn _mm256_cmpgt_epi64() {
        let a = i64x4::splat(0).replace(0, 5);
        let b = i64x4::splat(0);
        let r = avx2::_mm256_cmpgt_epi64(a, b);
        assert_eq!(r, i64x4::splat(0).replace(
            0, 0xFFFFFFFFFFFFFFFFu64 as i64));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hadd_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hadd_epi16(a, b);
        let e = i16x16::new(4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hadd_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_hadd_epi32(a, b);
        let e = i32x8::new(4, 4, 8, 8, 4, 4, 8, 8);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hadds_epi16() {
        let a = i16x16::splat(2).replace(0,0x7FFF).replace(1,1);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hadds_epi16(a, b);
        let e = i16x16::new(
            0x7FFF, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_hsub_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hsub_epi16(a, b);
        let e = i16x16::splat(0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hsub_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_hsub_epi32(a, b);
        let e = i32x8::splat(0);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hsubs_epi16() {
        let a = i16x16::splat(2).replace(0,0x7FFF).replace(1,-1);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hsubs_epi16(a, b);
        let e = i16x16::splat(0).replace(0,0x7FFF);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_madd_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_madd_epi16(a, b);
        let e = i32x8::splat(16);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_maddubs_epi16() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_maddubs_epi16(a, b);
        let e = i16x16::splat(16);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_max_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_max_epi16(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_max_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_max_epi32(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_max_epi8() {
        let a = i8x32::splat(2);
        let b = i8x32::splat(4);
        let r = avx2::_mm256_max_epi8(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_max_epu16() {
        let a = u16x16::splat(2);
        let b = u16x16::splat(4);
        let r = avx2::_mm256_max_epu16(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_max_epu32() {
        let a = u32x8::splat(2);
        let b = u32x8::splat(4);
        let r = avx2::_mm256_max_epu32(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_max_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_max_epu8(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_min_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_min_epi16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_min_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_min_epi32(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_min_epi8() {
        let a = i8x32::splat(2);
        let b = i8x32::splat(4);
        let r = avx2::_mm256_min_epi8(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_min_epu16() {
        let a = u16x16::splat(2);
        let b = u16x16::splat(4);
        let r = avx2::_mm256_min_epu16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_min_epu32() {
        let a = u32x8::splat(2);
        let b = u32x8::splat(4);
        let r = avx2::_mm256_min_epu32(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_min_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_min_epu8(a, b);
        assert_eq!(r, a);
    }


/**
    // TODO this fails in debug but not release, why?
    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_movemask_epi8() {
        let a = i8x32::splat(-1);
        let r = avx2::_mm256_movemask_epi8(a);
        let e : i32 = -1;
        assert_eq!(r, e);
    }

    // TODO This fails in debug but not in release, whhhy?
    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mpsadbw_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_mpsadbw_epu8(a, b, 0);
        let e = u16x16::splat(8);
        assert_eq!(r, e);
    }
**/

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mul_epi32() {
        let a = i32x8::new(0, 0, 0, 0, 2, 2, 2, 2);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_mul_epi32(a, b);
        let e = i64x4::new(0, 0, 10, 14);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mul_epu32() {
        let a = u32x8::new(0, 0, 0, 0, 2, 2, 2, 2);
        let b = u32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_mul_epu32(a, b);
        let e = u64x4::new(0, 0, 10, 14);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mulhi_epi16() {
        let a = i16x16::splat(6535);
        let b = i16x16::splat(6535);
        let r = avx2::_mm256_mulhi_epi16(a, b);
        let e = i16x16::splat(651);
        assert_eq!(r, e);
    }

      #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mulhi_epu16() {
        let a = u16x16::splat(6535);
        let b = u16x16::splat(6535);
        let r = avx2::_mm256_mulhi_epu16(a, b);
        let e = u16x16::splat(651);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mullo_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_mullo_epi16(a, b);
        let e = i16x16::splat(8);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mullo_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_mullo_epi32(a, b);
        let e = i32x8::splat(8);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_mulhrs_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_mullo_epi16(a, b);
        let e = i16x16::splat(8);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_or_si256() {
        let a = __m256i::splat(-1);
        let b = __m256i::splat(0);
        let r = avx2::_mm256_or_si256(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_packs_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_packs_epi16(a, b);
        let e = i8x32::new(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_packs_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_packs_epi32(a, b);
        let e = i16x16::new(
            2, 2, 2, 2,
            4, 4, 4, 4,
            2, 2, 2, 2,
            4, 4, 4, 4);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_packus_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_packus_epi16(a, b);
        let e = u8x32::new(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_packus_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_packus_epi32(a, b);
        let e = u16x16::new(
            2, 2, 2, 2,
            4, 4, 4, 4,
            2, 2, 2, 2,
            4, 4, 4, 4);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sad_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_sad_epu8(a, b);
        let e = u64x4::splat(16);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sign_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(-1);
        let r = avx2::_mm256_sign_epi16(a, b);
        let e = i16x16::splat(-2);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sign_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(-1);
        let r = avx2::_mm256_sign_epi32(a, b);
        let e = i32x8::splat(-2);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sign_epi8() {
        let a = i8x32::splat(2);
        let b = i8x32::splat(-1);
        let r = avx2::_mm256_sign_epi8(a, b);
        let e = i8x32::splat(-2);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sll_epi16() {
         assert_eq!(
            avx2::_mm256_sll_epi16(i16x16::splat(0xFF), i16x8::splat(0).replace(0,4)),
            i16x16::splat(0xFF0));

    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sll_epi32() {
         assert_eq!(
            avx2::_mm256_sll_epi32(i32x8::splat(0xFFFF), i32x4::splat(0).replace(0,4)),
            i32x8::splat(0xFFFF0));

    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sll_epi64() {
         assert_eq!(
            avx2::_mm256_sll_epi64(i64x4::splat(0xFFFFFFFF), i64x2::splat(0).replace(0,4)),
            i64x4::splat(0xFFFFFFFF0));

    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_slli_epi16() {
        assert_eq!(
            avx2::_mm256_slli_epi16(i16x16::splat(0xFF), 4),
            i16x16::splat(0xFF0));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_slli_epi32() {
        assert_eq!(
            avx2::_mm256_slli_epi32(i32x8::splat(0xFFFF), 4),
            i32x8::splat(0xFFFF0));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_slli_epi64() {
        assert_eq!(
            avx2::_mm256_slli_epi64(i64x4::splat(0xFFFFFFFF), 4),
            i64x4::splat(0xFFFFFFFF0));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm_sllv_epi32() {
        let a = i32x4::splat(2);
        let b = i32x4::splat(1);
        let r = avx2::_mm_sllv_epi32(a, b);
        let e = i32x4::splat(4);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sllv_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(1);
        let r = avx2::_mm256_sllv_epi32(a, b);
        let e = i32x8::splat(4);
        assert_eq!(r, e);
    }
    #[test]
    #[target_feature = "+avx2"]
    fn _mm_sllv_epi64() {
        let a = i64x2::splat(2);
        let b = i64x2::splat(1);
        let r = avx2::_mm_sllv_epi64(a, b);
        let e = i64x2::splat(4);
        assert_eq!(r, e);
    }
    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sllv_epi64() {
        let a = i64x4::splat(2);
        let b = i64x4::splat(1);
        let r = avx2::_mm256_sllv_epi64(a, b);
        let e = i64x4::splat(4);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_sra_epi16() {
         assert_eq!(
            avx2::_mm256_sra_epi16(
                i16x16::splat(-1), i16x8::new(1, 0, 0, 0, 0, 0, 0, 0)),
            i16x16::splat(-1));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_sra_epi32() {
         assert_eq!(
            avx2::_mm256_sra_epi32(
                i32x8::splat(-1), i32x4::splat(0).replace(0,1)),
            i32x8::splat(-1));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srai_epi16() {
           assert_eq!(
            avx2::_mm256_srai_epi16(
                i16x16::splat(-1), 1), i16x16::splat(-1));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srai_epi32() {
           assert_eq!(
            avx2::_mm256_srai_epi32(
                i32x8::splat(-1), 1), i32x8::splat(-1));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm_srav_epi32() {
        let a = i32x4::splat(4);
        let count = i32x4::splat(1);
        let r = avx2::_mm_srav_epi32(a, count);
        let e = i32x4::splat(2);
        assert_eq!(r, e );
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srav_epi32() {
        let a = i32x8::splat(4);
        let count = i32x8::splat(1);
        let r = avx2::_mm256_srav_epi32(a, count);
        let e = i32x8::splat(2);
        assert_eq!(r, e );
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srl_epi16() {
        assert_eq!(
            avx2::_mm256_srl_epi16(
                i16x16::splat(0xFF), i16x8::splat(0).replace(0,4)),
            i16x16::splat(0xF));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srl_epi32() {
        assert_eq!(
            avx2::_mm256_srl_epi32(
                i32x8::splat(0xFFFF), i32x4::splat(0).replace(0,4)),
            i32x8::splat(0xFFF));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srl_epi64() {
        assert_eq!(
            avx2::_mm256_srl_epi64(
                i64x4::splat(0xFFFFFFFF), i64x2::splat(0).replace(0,4)),
            i64x4::splat(0xFFFFFFF));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srli_epi16() {
        assert_eq!(
            avx2::_mm256_srli_epi16(i16x16::splat(0xFF), 4),
            i16x16::splat(0xF));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srli_epi32() {
        assert_eq!(
            avx2::_mm256_srli_epi32(i32x8::splat(0xFFFF), 4),
            i32x8::splat(0xFFF));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_srli_epi64() {
        assert_eq!(
            avx2::_mm256_srli_epi64(i64x4::splat(0xFFFFFFFF), 4),
            i64x4::splat(0xFFFFFFF));
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm_srlv_epi32() {
        let a = i32x4::splat(2);
        let count = i32x4::splat(1);
        let r = avx2::_mm_srlv_epi32(a, count);
        let e = i32x4::splat(1);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_srlv_epi32() {
        let a = i32x8::splat(2);
        let count = i32x8::splat(1);
        let r = avx2::_mm256_srlv_epi32(a, count);
        let e = i32x8::splat(1);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm_srlv_epi64() {
        let a = i64x2::splat(2);
        let count = i64x2::splat(1);
        let r = avx2::_mm_srlv_epi64(a, count);
        let e = i64x2::splat(1);
        assert_eq!(r, e);
    }


    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_srlv_epi64() {
        let a = i64x4::splat(2);
        let count = i64x4::splat(1);
        let r = avx2::_mm256_srlv_epi64(a, count);
        let e = i64x4::splat(1);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sub_epi16() {
        let a = i16x16::splat(4);
        let b = i16x16::splat(2);
        let r = avx2::_mm256_sub_epi16(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sub_epi32() {
        let a = i32x8::splat(4);
        let b = i32x8::splat(2);
        let r = avx2::_mm256_sub_epi32(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sub_epi64() {
        let a = i64x4::splat(4);
        let b = i64x4::splat(2);
        let r = avx2::_mm256_sub_epi64(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_sub_epi8() {
        let a = i8x32::splat(4);
        let b = i8x32::splat(2);
        let r = avx2::_mm256_sub_epi8(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_subs_epi16() {
        let a = i16x16::splat(4);
        let b = i16x16::splat(2);
        let r = avx2::_mm256_subs_epi16(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_subs_epi8() {
        let a = i8x32::splat(4);
        let b = i8x32::splat(2);
        let r = avx2::_mm256_subs_epi8(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_subs_epu16() {
        let a = u16x16::splat(4);
        let b = u16x16::splat(2);
        let r = avx2::_mm256_subs_epu16(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_subs_epu8() {
        let a = u8x32::splat(4);
        let b = u8x32::splat(2);
        let r = avx2::_mm256_subs_epu8(a, b);
        assert_eq!(r, b);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_xor_si256() {
        assert_eq!(
            avx2::_mm256_xor_si256(__m256i::splat(5), __m256i::splat(3)),
            __m256i::splat(6));
    }


}

pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from an XML specification, published under the same license as the
// `intrinsic-test` crate.\n";

pub const PLATFORM_RUST_DEFINITIONS: &str = r#"
use core_arch::arch::x86_64::*;

#[inline]
unsafe fn _mm_loadu_ph_to___m128i(mem_addr: *const f16) -> __m128i {
    _mm_castph_si128(_mm_loadu_ph(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_ph_to___m256i(mem_addr: *const f16) -> __m256i {
    _mm256_castph_si256(_mm256_loadu_ph(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_ph_to___mm512i(mem_addr: *const f16) -> __m512i {
    _mm512_castph_si512(_mm512_loadu_ph(mem_addr))
}


#[inline]
unsafe fn _mm_loadu_ps_to___m128h(mem_addr: *const f32) -> __m128h {
    _mm_castps_ph(_mm_loadu_ps(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_ps_to___m256h(mem_addr: *const f32) -> __m256h {
    _mm256_castps_ph(_mm256_loadu_ps(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_ps_to___m512h(mem_addr: *const f32) -> __m512h {
    _mm512_castps_ph(_mm512_loadu_ps(mem_addr))
}

#[inline]
unsafe fn _mm_loadu_epi16_to___m128d(mem_addr: *const i16) -> __m128d {
    _mm_castsi128_pd(_mm_loadu_epi16(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_epi16_to___m256d(mem_addr: *const i16) -> __m256d {
    _mm256_castsi256_pd(_mm256_loadu_epi16(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_epi16_to___m512d(mem_addr: *const i16) -> __m512d {
    _mm512_castsi512_pd(_mm512_loadu_epi16(mem_addr))
}

#[inline]
unsafe fn _mm_loadu_epi32_to___m128d(mem_addr: *const i32) -> __m128d {
    _mm_castsi128_pd(_mm_loadu_epi32(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_epi32_to___m256d(mem_addr: *const i32) -> __m256d {
    _mm256_castsi256_pd(_mm256_loadu_epi32(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_epi32_to___m512d(mem_addr: *const i32) -> __m512d {
    _mm512_castsi512_pd(_mm512_loadu_epi32(mem_addr))
}

#[inline]
unsafe fn _mm_loadu_epi64_to___m128d(mem_addr: *const i64) -> __m128d {
    _mm_castsi128_pd(_mm_loadu_epi64(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_epi64_to___m256d(mem_addr: *const i64) -> __m256d {
    _mm256_castsi256_pd(_mm256_loadu_epi64(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_epi64_to___m512d(mem_addr: *const i64) -> __m512d {
    _mm512_castsi512_pd(_mm512_loadu_epi64(mem_addr))
}

// === 
#[inline]
unsafe fn _mm_loadu_epi16_to___m128(mem_addr: *const i16) -> __m128 {
    _mm_castsi128_ps(_mm_loadu_epi16(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_epi16_to___m256(mem_addr: *const i16) -> __m256 {
    _mm256_castsi256_ps(_mm256_loadu_epi16(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_epi16_to___m512(mem_addr: *const i16) -> __m512 {
    _mm512_castsi512_ps(_mm512_loadu_epi16(mem_addr))
}

#[inline]
unsafe fn _mm_loadu_epi32_to___m128(mem_addr: *const i32) -> __m128 {
    _mm_castsi128_ps(_mm_loadu_epi32(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_epi32_to___m256(mem_addr: *const i32) -> __m256 {
    _mm256_castsi256_ps(_mm256_loadu_epi32(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_epi32_to___m512(mem_addr: *const i32) -> __m512 {
    _mm512_castsi512_ps(_mm512_loadu_epi32(mem_addr))
}

#[inline]
unsafe fn _mm_loadu_epi64_to___m128(mem_addr: *const i64) -> __m128 {
    _mm_castsi128_ps(_mm_loadu_epi64(mem_addr))
}

#[inline]
unsafe fn _mm256_loadu_epi64_to___m256(mem_addr: *const i64) -> __m256 {
    _mm256_castsi256_ps(_mm256_loadu_epi64(mem_addr))
}

#[inline]
unsafe fn _mm512_loadu_epi64_to___m512(mem_addr: *const i64) -> __m512 {
    _mm512_castsi512_ps(_mm512_loadu_epi64(mem_addr))
}

"#;

pub const PLATFORM_RUST_CFGS: &str = r#"
#![feature(stdarch_x86_avx512_bf16)]
#![feature(stdarch_x86_avx512_f16)]
#![feature(stdarch_x86_rtm)]
#![feature(x86_amx_intrinsics)]
"#;

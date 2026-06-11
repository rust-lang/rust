mod constraint;
mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitecture;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use intrinsic::X86IntrinsicType;
use xml_parser::get_xml_intrinsics;

pub struct X86 {
    intrinsics: Vec<Intrinsic<X86>>,
}

impl SupportedArchitecture for X86 {
    type Type = X86IntrinsicType;

    fn intrinsics(&self) -> &[Intrinsic<Self>] {
        &self.intrinsics
    }

    const NOTICE: &str = r#"
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from an XML specification, published under the same license as the
// `intrinsic-test` crate.
"#;

    const C_PRELUDE: &str = r#"
#include <immintrin.h>
"#;
    const RUST_PRELUDE: &str = RUST_PRELUDE;

    fn c_compiler_flags(&self, _cli_options: &ProcessedCli) -> Vec<&str> {
        vec![
            "-maes",
            "-mf16c",
            "-mfma",
            "-mavx",
            "-mavx2",
            "-mavx512f",
            "-msse2",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512cd",
            "-mavx512fp16",
            "-msha",
            "-msha512",
            "-msm3",
            "-msm4",
            "-mavxvnni",
            "-mavxvnniint8",
            "-mavxneconvert",
            "-mavxifma",
            "-mavxvnniint16",
            "-mavx512bf16",
            "-mavx512bitalg",
            "-mavx512ifma",
            "-mavx512vbmi",
            "-mavx512vbmi2",
            "-mavx512vnni",
            "-mavx512vpopcntdq",
            "-mavx512vp2intersect",
            "-mbmi",
            "-mbmi2",
            "-mgfni",
            "-mvaes",
            "-mvpclmulqdq",
            "-mlzcnt",
        ]
    }

    fn create(cli_options: &ProcessedCli) -> Self {
        let mut intrinsics =
            get_xml_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        intrinsics.dedup_by(|a, b| a.name == b.name);

        let sample_percentage: usize = cli_options.sample_percentage as usize;
        let sample_size = (intrinsics.len() * sample_percentage) / 100;

        let intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| i.arguments.args.len() > 0)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .take(sample_size)
            .collect::<Vec<_>>();

        Self { intrinsics }
    }
}

const RUST_PRELUDE: &str = r#"
#![feature(stdarch_x86_avx512_bf16)]
#![feature(stdarch_x86_avx512_f16)]
#![feature(stdarch_x86_rtm)]
#![feature(x86_amx_intrinsics)]

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

pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from an XML specification, published under the same license as the
// `intrinsic-test` crate.\n";

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
#[repr(transparent)]
struct Hex<T>(T);
 "#;

pub const LANE_FUNCTION_HELPERS: &str = r#"
typedef float float16_t;
typedef float float32_t;
typedef double float64_t;

#define __int64 long long
#define __int32 int

#define _mm512_extract_intrinsic_test_epi8(m, lane) \
    _mm_extract_epi8(_mm512_extracti64x2_epi64((m), (lane) / 16), (lane) % 16)

#define _mm512_extract_intrinsic_test_epi16(m, lane) \
    _mm_extract_epi16(_mm512_extracti64x2_epi64((m), (lane) / 8), (lane) % 8)

#define _mm512_extract_intrinsic_test_epi32(m, lane) \
    _mm_extract_epi32(_mm512_extracti64x2_epi64((m), (lane) / 4), (lane) % 4)

#define _mm512_extract_intrinsic_test_epi64(m, lane) \
    _mm_extract_epi64(_mm512_extracti64x2_epi64((m), (lane) / 2), (lane) % 2)

#define _mm64_extract_intrinsic_test_epi8(m, lane) \
    ((_mm_extract_pi16((m), (lane) / 2) >> (((lane) % 2) * 8)) & 0xFF)

#define _mm64_extract_intrinsic_test_epi32(m, lane) \
    _mm_cvtsi64_si32(_mm_srli_si64(m, (lane) * 32))
"#;

pub const X86_CONFIGURATIONS: &str = r#"
#![cfg_attr(target_arch = "x86", feature(avx))]
#![cfg_attr(target_arch = "x86", feature(sse))]
#![cfg_attr(target_arch = "x86", feature(sse2))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512_bf16))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512_f16))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_rtm))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_rtm))]
#![cfg_attr(target_arch = "x86_64", feature(sse))]
#![cfg_attr(target_arch = "x86_64", feature(sse2))]
#![cfg_attr(target_arch = "x86_64", feature(x86_amx_intrinsics))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]
#![feature(fmt_helpers_for_derive)]
"#;

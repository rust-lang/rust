pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from an XML specification, published under the same license as the
// `intrinsic-test` crate.\n";

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
use std::arch::x86_64::*;

#[inline]
fn debug_simd_finish<T: core::fmt::Debug, const N: usize>(
    formatter: &mut core::fmt::Formatter<'_>,
    type_name: &str,
    array: &[T; N],
) -> core::fmt::Result {
    core::fmt::Formatter::debug_tuple_fields_finish(
        formatter,
        type_name,
        &core::array::from_fn::<&dyn core::fmt::Debug, N, _>(|i| &array[i]),
    )
}

#[repr(transparent)]
struct Hex<T>(T);

impl<T: DebugHexF16> core::fmt::Debug for Hex<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <T as DebugHexF16>::fmt(&self.0, f)
    }
}

fn debug_f16<T: DebugHexF16>(x: T) -> impl core::fmt::Debug {
    Hex(x)
}

trait DebugHexF16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;
}

impl DebugHexF16 for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:#06x?}", self.to_bits())
    }
}

impl DebugHexF16 for __m128h {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 8]>(*self) };
        debug_simd_finish(f, "__m128h", &array)
    }
}

impl DebugHexF16 for __m256h {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 16]>(*self) };
        debug_simd_finish(f, "__m256h", &array)
    }
}

impl DebugHexF16 for __m512h {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 32]>(*self) };
        debug_simd_finish(f, "__m512h", &array)
    }
}
 "#;

pub const LANE_FUNCTION_HELPERS: &str = r#"
typedef _Float16 float16_t;
typedef float float32_t;
typedef double float64_t;

#define __int64 long long
#define __int32 int

std::ostream& operator<<(std::ostream& os, _Float16 value);
std::ostream& operator<<(std::ostream& os, __m128i value);
std::ostream& operator<<(std::ostream& os, __m256i value);
std::ostream& operator<<(std::ostream& os, __m512i value);

std::ostream& operator<<(std::ostream& os, _Float16 value) {
    uint16_t temp = 0;
    memcpy(&temp, &value, sizeof(_Float16));
    std::stringstream ss;
    ss << "0x" << std::setfill('0') << std::setw(4) << std::hex << temp;
    os << ss.str();
    return os;
}

std::ostream& operator<<(std::ostream& os, __m128i value) {
    void* temp = malloc(sizeof(__m128i));
    _mm_storeu_si128((__m128i*)temp, value);
    std::stringstream ss;
    
    ss << "0x";
    for(int i = 0; i < 16; i++) {
        ss << std::setfill('0') << std::setw(2) << std::hex << ((char*)temp)[i];
    }
    os << ss.str();
    return os;
}

std::ostream& operator<<(std::ostream& os, __m256i value) {
    void* temp = malloc(sizeof(__m256i));
    _mm256_storeu_si256((__m256i*)temp, value);
    std::stringstream ss;
    
    ss << "0x";
    for(int i = 0; i < 32; i++) {
        ss << std::setfill('0') << std::setw(2) << std::hex << ((char*)temp)[i];
    }
    os << ss.str();
    return os;
}

std::ostream& operator<<(std::ostream& os, __m512i value) {
    void* temp = malloc(sizeof(__m512i));
    _mm512_storeu_si512((__m512i*)temp, value);
    std::stringstream ss;
    
    ss << "0x";
    for(int i = 0; i < 64; i++) {
        ss << std::setfill('0') << std::setw(2) << std::hex << ((char*)temp)[i];
    }
    os << ss.str();
    return os;
}

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
#![cfg_attr(target_arch = "x86_64", feature(x86_amx_intrinsics))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]
#![feature(fmt_helpers_for_derive)]
"#;

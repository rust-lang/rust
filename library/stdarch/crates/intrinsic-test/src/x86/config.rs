pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from an XML specification, published under the same license as the
// `intrinsic-test` crate.\n";

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const PLATFORM_RUST_DEFINITIONS: &str = r#"
use std::arch::x86_64::*;

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

impl DebugHexF16 for __m128i {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 8]>(*self) };
        debug_simd_finish(f, "__m128i", &array)
    }
}

impl DebugHexF16 for __m256h {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 16]>(*self) };
        debug_simd_finish(f, "__m256h", &array)
    }
}

impl DebugHexF16 for __m256i {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 16]>(*self) };
        debug_simd_finish(f, "__m256i", &array)
    }
}

impl DebugHexF16 for __m512h {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 32]>(*self) };
        debug_simd_finish(f, "__m512h", &array)
    }
}

impl DebugHexF16 for __m512i {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 32]>(*self) };
        debug_simd_finish(f, "__m512i", &array)
    }
}

trait DebugAs<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;
}

impl<T: core::fmt::Display> DebugAs<T> for T {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

macro_rules! impl_debug_as {
    ($simd:ty, $name:expr, $bits:expr, [$($type:ty),+]) => {
        $(
            impl DebugAs<$type> for $simd {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    const ELEMENT_BITS: usize = core::mem::size_of::<$type>() * 8;
                    const NUM_ELEMENTS: usize = $bits / ELEMENT_BITS;
                    let array = unsafe { core::mem::transmute::<_, [$type; NUM_ELEMENTS]>(*self) };
                    debug_simd_finish(f, $name, &array)
                }
            }
        )+
    };
}

impl_debug_as!(__m128i, "__m128i", 128, [u8, i8, u16, i16, u32, i32, u64, i64]);
impl_debug_as!(__m256i, "__m256i", 256, [u8, i8, u16, i16, u32, i32, u64, i64]);
impl_debug_as!(__m512i, "__m512i", 512, [u8, i8, u16, i16, u32, i32, u64, i64]);
impl_debug_as!(__m128h, "__m128h", 128, [f32]);
impl_debug_as!(__m256h, "__m256h", 256, [f32]);
impl_debug_as!(__m512h, "__m512h", 512, [f32]);

fn debug_as<V, T>(x: V) -> impl core::fmt::Debug 
where V: DebugAs<T>
{
    struct DebugWrapper<V, T>(V, core::marker::PhantomData<T>);
    impl<V: DebugAs<T>, T> core::fmt::Debug for DebugWrapper<V, T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.fmt(f)
        }
    }
    DebugWrapper(x, core::marker::PhantomData)
}

"#;

pub const PLATFORM_C_FORWARD_DECLARATIONS: &str = r#"
#ifndef X86_DECLARATIONS
#define X86_DECLARATIONS
    typedef _Float16 float16_t;
    typedef float float32_t;
    typedef double float64_t;
    
    #define __int64 long long
    #define __int32 int

    std::ostream& operator<<(std::ostream& os, _Float16 value);
    std::ostream& operator<<(std::ostream& os, __m128i value);
    std::ostream& operator<<(std::ostream& os, __m256i value);
    std::ostream& operator<<(std::ostream& os, __m512i value);
    std::ostream& operator<<(std::ostream& os, __mmask8 value);
    
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
        
    // Load f16 (__m128h) and cast to integer (__m128i)
    #define _mm_loadu_ph_to___m128i(mem_addr) _mm_castph_si128(_mm_loadu_ph(mem_addr))
    #define _mm256_loadu_ph_to___m256i(mem_addr) _mm256_castph_si256(_mm256_loadu_ph(mem_addr))
    #define _mm512_loadu_ph_to___m512i(mem_addr) _mm512_castph_si512(_mm512_loadu_ph(mem_addr))
    
    // Load f32 (__m128) and cast to f16 (__m128h)
    #define _mm_loadu_ps_to___m128h(mem_addr) _mm_castps_ph(_mm_loadu_ps(mem_addr))
    #define _mm256_loadu_ps_to___m256h(mem_addr) _mm256_castps_ph(_mm256_loadu_ps(mem_addr))
    #define _mm512_loadu_ps_to___m512h(mem_addr) _mm512_castps_ph(_mm512_loadu_ps(mem_addr))
    
    // Load integer types and cast to double (__m128d, __m256d, __m512d)
    #define _mm_loadu_epi16_to___m128d(mem_addr) _mm_castsi128_pd(_mm_loadu_si128((__m128i const*)(mem_addr)))
    #define _mm256_loadu_epi16_to___m256d(mem_addr) _mm256_castsi256_pd(_mm256_loadu_si256((__m256i const*)(mem_addr)))
    #define _mm512_loadu_epi16_to___m512d(mem_addr) _mm512_castsi512_pd(_mm512_loadu_si512((__m512i const*)(mem_addr)))
    
    #define _mm_loadu_epi32_to___m128d(mem_addr) _mm_castsi128_pd(_mm_loadu_si128((__m128i const*)(mem_addr)))
    #define _mm256_loadu_epi32_to___m256d(mem_addr) _mm256_castsi256_pd(_mm256_loadu_si256((__m256i const*)(mem_addr)))
    #define _mm512_loadu_epi32_to___m512d(mem_addr) _mm512_castsi512_pd(_mm512_loadu_si512((__m512i const*)(mem_addr)))
    
    #define _mm_loadu_epi64_to___m128d(mem_addr) _mm_castsi128_pd(_mm_loadu_si128((__m128i const*)(mem_addr)))
    #define _mm256_loadu_epi64_to___m256d(mem_addr) _mm256_castsi256_pd(_mm256_loadu_si256((__m256i const*)(mem_addr)))
    #define _mm512_loadu_epi64_to___m512d(mem_addr) _mm512_castsi512_pd(_mm512_loadu_si512((__m512i const*)(mem_addr)))
    
    // Load integer types and cast to float (__m128, __m256, __m512)
    #define _mm_loadu_epi16_to___m128(mem_addr) _mm_castsi128_ps(_mm_loadu_si128((__m128i const*)(mem_addr)))
    #define _mm256_loadu_epi16_to___m256(mem_addr) _mm256_castsi256_ps(_mm256_loadu_si256((__m256i const*)(mem_addr)))
    #define _mm512_loadu_epi16_to___m512(mem_addr) _mm512_castsi512_ps(_mm512_loadu_si512((__m512i const*)(mem_addr)))
    
    #define _mm_loadu_epi32_to___m128(mem_addr) _mm_castsi128_ps(_mm_loadu_si128((__m128i const*)(mem_addr)))
    #define _mm256_loadu_epi32_to___m256(mem_addr) _mm256_castsi256_ps(_mm256_loadu_si256((__m256i const*)(mem_addr)))
    #define _mm512_loadu_epi32_to___m512(mem_addr) _mm512_castsi512_ps(_mm512_loadu_si512((__m512i const*)(mem_addr)))
    
    #define _mm_loadu_epi64_to___m128(mem_addr) _mm_castsi128_ps(_mm_loadu_si128((__m128i const*)(mem_addr)))
    #define _mm256_loadu_epi64_to___m256(mem_addr) _mm256_castsi256_ps(_mm256_loadu_si256((__m256i const*)(mem_addr)))
    #define _mm512_loadu_epi64_to___m512(mem_addr) _mm512_castsi512_ps(_mm512_loadu_si512((__m512i const*)(mem_addr)))
    
    // T1 is the `To` type, T2 is the `From` type
    template<typename T1, typename T2> T1 cast(T2 x) {
      if constexpr ((std::is_integral_v<T1> && std::is_integral_v<T2>) || (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>)) {
          return x;
      } else if constexpr (sizeof(T1) <= sizeof(T2)) {
        T1 ret{};
        std::memcpy(&ret, &x, sizeof(T1));
        return ret;
      } else {
        static_assert(sizeof(T1) == sizeof(T2) || std::is_convertible_v<T2, T1>,
                              "T2 must either be convertible to T1, or have the same size as T1!");
        return T1{};
      }
    }
#endif
"#;
pub const PLATFORM_C_DEFINITIONS: &str = r#"

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

std::ostream& operator<<(std::ostream& os, __mmask8 value) {
    os << static_cast<int>(value);
    return os;
}
"#;

pub const PLATFORM_RUST_CFGS: &str = r#"
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

//! `x86` and `x86_64` intrinsics.

use crate::{intrinsics, marker::Sized, mem::transmute};

#[macro_use]
mod macros;

types! {
    /// 128-bit wide integer vector type, x86-specific
    ///
    /// This type is the same as the `__m128i` type defined by Intel,
    /// representing a 128-bit SIMD register. Usage of this type typically
    /// corresponds to the `sse` and up target features for x86/x86_64.
    ///
    /// Internally this type may be viewed as:
    ///
    /// * `i8x16` - sixteen `i8` variables packed together
    /// * `i16x8` - eight `i16` variables packed together
    /// * `i32x4` - four `i32` variables packed together
    /// * `i64x2` - two `i64` variables packed together
    ///
    /// (as well as unsigned versions). Each intrinsic may interpret the
    /// internal bits differently, check the documentation of the intrinsic
    /// to see how it's being used.
    ///
    /// Note that this means that an instance of `__m128i` typically just means
    /// a "bag of bits" which is left up to interpretation at the point of use.
    ///
    /// Most intrinsics using `__m128i` are prefixed with `_mm_` and the
    /// integer types tend to correspond to suffixes like "epi8" or "epi32".
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(target_arch = "x86")]
    /// use std::arch::x86::*;
    /// #[cfg(target_arch = "x86_64")]
    /// use std::arch::x86_64::*;
    ///
    /// # fn main() {
    /// # #[target_feature(enable = "sse2")]
    /// # unsafe fn foo() {
    /// let all_bytes_zero = _mm_setzero_si128();
    /// let all_bytes_one = _mm_set1_epi8(1);
    /// let four_i32 = _mm_set_epi32(1, 2, 3, 4);
    /// # }
    /// # if is_x86_feature_detected!("sse2") { unsafe { foo() } }
    /// # }
    /// ```
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub struct __m128i(i64, i64);

    /// 128-bit wide set of four `f32` types, x86-specific
    ///
    /// This type is the same as the `__m128` type defined by Intel,
    /// representing a 128-bit SIMD register which internally is consisted of
    /// four packed `f32` instances. Usage of this type typically corresponds
    /// to the `sse` and up target features for x86/x86_64.
    ///
    /// Note that unlike `__m128i`, the integer version of the 128-bit
    /// registers, this `__m128` type has *one* interpretation. Each instance
    /// of `__m128` always corresponds to `f32x4`, or four `f32` types packed
    /// together.
    ///
    /// Most intrinsics using `__m128` are prefixed with `_mm_` and are
    /// suffixed with "ps" (or otherwise contain "ps"). Not to be confused with
    /// "pd" which is used for `__m128d`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(target_arch = "x86")]
    /// use std::arch::x86::*;
    /// #[cfg(target_arch = "x86_64")]
    /// use std::arch::x86_64::*;
    ///
    /// # fn main() {
    /// # #[target_feature(enable = "sse")]
    /// # unsafe fn foo() {
    /// let four_zeros = _mm_setzero_ps();
    /// let four_ones = _mm_set1_ps(1.0);
    /// let four_floats = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    /// # }
    /// # if is_x86_feature_detected!("sse") { unsafe { foo() } }
    /// # }
    /// ```
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub struct __m128(f32, f32, f32, f32);

    /// 128-bit wide set of two `f64` types, x86-specific
    ///
    /// This type is the same as the `__m128d` type defined by Intel,
    /// representing a 128-bit SIMD register which internally is consisted of
    /// two packed `f64` instances. Usage of this type typically corresponds
    /// to the `sse` and up target features for x86/x86_64.
    ///
    /// Note that unlike `__m128i`, the integer version of the 128-bit
    /// registers, this `__m128d` type has *one* interpretation. Each instance
    /// of `__m128d` always corresponds to `f64x2`, or two `f64` types packed
    /// together.
    ///
    /// Most intrinsics using `__m128d` are prefixed with `_mm_` and are
    /// suffixed with "pd" (or otherwise contain "pd"). Not to be confused with
    /// "ps" which is used for `__m128`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(target_arch = "x86")]
    /// use std::arch::x86::*;
    /// #[cfg(target_arch = "x86_64")]
    /// use std::arch::x86_64::*;
    ///
    /// # fn main() {
    /// # #[target_feature(enable = "sse")]
    /// # unsafe fn foo() {
    /// let two_zeros = _mm_setzero_pd();
    /// let two_ones = _mm_set1_pd(1.0);
    /// let two_floats = _mm_set_pd(1.0, 2.0);
    /// # }
    /// # if is_x86_feature_detected!("sse") { unsafe { foo() } }
    /// # }
    /// ```
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub struct __m128d(f64, f64);

    /// 256-bit wide integer vector type, x86-specific
    ///
    /// This type is the same as the `__m256i` type defined by Intel,
    /// representing a 256-bit SIMD register. Usage of this type typically
    /// corresponds to the `avx` and up target features for x86/x86_64.
    ///
    /// Internally this type may be viewed as:
    ///
    /// * `i8x32` - thirty two `i8` variables packed together
    /// * `i16x16` - sixteen `i16` variables packed together
    /// * `i32x8` - eight `i32` variables packed together
    /// * `i64x4` - four `i64` variables packed together
    ///
    /// (as well as unsigned versions). Each intrinsic may interpret the
    /// internal bits differently, check the documentation of the intrinsic
    /// to see how it's being used.
    ///
    /// Note that this means that an instance of `__m256i` typically just means
    /// a "bag of bits" which is left up to interpretation at the point of use.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(target_arch = "x86")]
    /// use std::arch::x86::*;
    /// #[cfg(target_arch = "x86_64")]
    /// use std::arch::x86_64::*;
    ///
    /// # fn main() {
    /// # #[target_feature(enable = "avx")]
    /// # unsafe fn foo() {
    /// let all_bytes_zero = _mm256_setzero_si256();
    /// let all_bytes_one = _mm256_set1_epi8(1);
    /// let eight_i32 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    /// # }
    /// # if is_x86_feature_detected!("avx") { unsafe { foo() } }
    /// # }
    /// ```
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub struct __m256i(i64, i64, i64, i64);

    /// 256-bit wide set of eight `f32` types, x86-specific
    ///
    /// This type is the same as the `__m256` type defined by Intel,
    /// representing a 256-bit SIMD register which internally is consisted of
    /// eight packed `f32` instances. Usage of this type typically corresponds
    /// to the `avx` and up target features for x86/x86_64.
    ///
    /// Note that unlike `__m256i`, the integer version of the 256-bit
    /// registers, this `__m256` type has *one* interpretation. Each instance
    /// of `__m256` always corresponds to `f32x8`, or eight `f32` types packed
    /// together.
    ///
    /// Most intrinsics using `__m256` are prefixed with `_mm256_` and are
    /// suffixed with "ps" (or otherwise contain "ps"). Not to be confused with
    /// "pd" which is used for `__m256d`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(target_arch = "x86")]
    /// use std::arch::x86::*;
    /// #[cfg(target_arch = "x86_64")]
    /// use std::arch::x86_64::*;
    ///
    /// # fn main() {
    /// # #[target_feature(enable = "avx")]
    /// # unsafe fn foo() {
    /// let eight_zeros = _mm256_setzero_ps();
    /// let eight_ones = _mm256_set1_ps(1.0);
    /// let eight_floats = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// # }
    /// # if is_x86_feature_detected!("avx") { unsafe { foo() } }
    /// # }
    /// ```
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub struct __m256(f32, f32, f32, f32, f32, f32, f32, f32);

    /// 256-bit wide set of four `f64` types, x86-specific
    ///
    /// This type is the same as the `__m256d` type defined by Intel,
    /// representing a 256-bit SIMD register which internally is consisted of
    /// four packed `f64` instances. Usage of this type typically corresponds
    /// to the `avx` and up target features for x86/x86_64.
    ///
    /// Note that unlike `__m256i`, the integer version of the 256-bit
    /// registers, this `__m256d` type has *one* interpretation. Each instance
    /// of `__m256d` always corresponds to `f64x4`, or four `f64` types packed
    /// together.
    ///
    /// Most intrinsics using `__m256d` are prefixed with `_mm256_` and are
    /// suffixed with "pd" (or otherwise contain "pd"). Not to be confused with
    /// "ps" which is used for `__m256`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(target_arch = "x86")]
    /// use std::arch::x86::*;
    /// #[cfg(target_arch = "x86_64")]
    /// use std::arch::x86_64::*;
    ///
    /// # fn main() {
    /// # #[target_feature(enable = "avx")]
    /// # unsafe fn foo() {
    /// let four_zeros = _mm256_setzero_pd();
    /// let four_ones = _mm256_set1_pd(1.0);
    /// let four_floats = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    /// # }
    /// # if is_x86_feature_detected!("avx") { unsafe { foo() } }
    /// # }
    /// ```
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub struct __m256d(f64, f64, f64, f64);

    /// 512-bit wide integer vector type, x86-specific
    ///
    /// This type is the same as the `__m512i` type defined by Intel,
    /// representing a 512-bit SIMD register. Usage of this type typically
    /// corresponds to the `avx512*` and up target features for x86/x86_64.
    ///
    /// Internally this type may be viewed as:
    ///
    /// * `i8x64` - sixty-four `i8` variables packed together
    /// * `i16x32` - thirty-two `i16` variables packed together
    /// * `i32x16` - sixteen `i32` variables packed together
    /// * `i64x8` - eight `i64` variables packed together
    ///
    /// (as well as unsigned versions). Each intrinsic may interpret the
    /// internal bits differently, check the documentation of the intrinsic
    /// to see how it's being used.
    ///
    /// Note that this means that an instance of `__m512i` typically just means
    /// a "bag of bits" which is left up to interpretation at the point of use.
    pub struct __m512i(i64, i64, i64, i64, i64, i64, i64, i64);

    /// 512-bit wide set of sixteen `f32` types, x86-specific
    ///
    /// This type is the same as the `__m512` type defined by Intel,
    /// representing a 512-bit SIMD register which internally is consisted of
    /// eight packed `f32` instances. Usage of this type typically corresponds
    /// to the `avx512*` and up target features for x86/x86_64.
    ///
    /// Note that unlike `__m512i`, the integer version of the 512-bit
    /// registers, this `__m512` type has *one* interpretation. Each instance
    /// of `__m512` always corresponds to `f32x16`, or sixteen `f32` types
    /// packed together.
    ///
    /// Most intrinsics using `__m512` are prefixed with `_mm512_` and are
    /// suffixed with "ps" (or otherwise contain "ps"). Not to be confused with
    /// "pd" which is used for `__m512d`.
    pub struct __m512(
        f32, f32, f32, f32, f32, f32, f32, f32,
        f32, f32, f32, f32, f32, f32, f32, f32,
    );

    /// 512-bit wide set of eight `f64` types, x86-specific
    ///
    /// This type is the same as the `__m512d` type defined by Intel,
    /// representing a 512-bit SIMD register which internally is consisted of
    /// eight packed `f64` instances. Usage of this type typically corresponds
    /// to the `avx` and up target features for x86/x86_64.
    ///
    /// Note that unlike `__m512i`, the integer version of the 512-bit
    /// registers, this `__m512d` type has *one* interpretation. Each instance
    /// of `__m512d` always corresponds to `f64x4`, or eight `f64` types packed
    /// together.
    ///
    /// Most intrinsics using `__m512d` are prefixed with `_mm512_` and are
    /// suffixed with "pd" (or otherwise contain "pd"). Not to be confused with
    /// "ps" which is used for `__m512`.
    pub struct __m512d(f64, f64, f64, f64, f64, f64, f64, f64);
}

/// The `__mmask16` type used in AVX-512 intrinsics, a 16-bit integer
#[allow(non_camel_case_types)]
pub type __mmask16 = u16;

/// The `__mmask8` type used in AVX-512 intrinsics, a 8-bit integer
#[allow(non_camel_case_types)]
pub type __mmask8 = u8;

/// The `_MM_CMPINT_ENUM` type used to specify comparison operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
pub type _MM_CMPINT_ENUM = i32;

/// The `MM_MANTISSA_NORM_ENUM` type used to specify mantissa normalized operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
pub type _MM_MANTISSA_NORM_ENUM = i32;

/// The `MM_MANTISSA_SIGN_ENUM` type used to specify mantissa signed operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
pub type _MM_MANTISSA_SIGN_ENUM = i32;

/// The `MM_PERM_ENUM` type used to specify shuffle operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
pub type _MM_PERM_ENUM = i32;

#[cfg(test)]
mod test;
#[cfg(test)]
pub use self::test::*;

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m128iExt: Sized {
    fn as_m128i(self) -> __m128i;

    #[inline]
    fn as_u8x16(self) -> crate::core_arch::simd::u8x16 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_u16x8(self) -> crate::core_arch::simd::u16x8 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_u32x4(self) -> crate::core_arch::simd::u32x4 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_u64x2(self) -> crate::core_arch::simd::u64x2 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_i8x16(self) -> crate::core_arch::simd::i8x16 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_i16x8(self) -> crate::core_arch::simd::i16x8 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_i32x4(self) -> crate::core_arch::simd::i32x4 {
        unsafe { transmute(self.as_m128i()) }
    }

    #[inline]
    fn as_i64x2(self) -> crate::core_arch::simd::i64x2 {
        unsafe { transmute(self.as_m128i()) }
    }
}

impl m128iExt for __m128i {
    #[inline]
    fn as_m128i(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m256iExt: Sized {
    fn as_m256i(self) -> __m256i;

    #[inline]
    fn as_u8x32(self) -> crate::core_arch::simd::u8x32 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_u16x16(self) -> crate::core_arch::simd::u16x16 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_u32x8(self) -> crate::core_arch::simd::u32x8 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_u64x4(self) -> crate::core_arch::simd::u64x4 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_i8x32(self) -> crate::core_arch::simd::i8x32 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_i16x16(self) -> crate::core_arch::simd::i16x16 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_i32x8(self) -> crate::core_arch::simd::i32x8 {
        unsafe { transmute(self.as_m256i()) }
    }

    #[inline]
    fn as_i64x4(self) -> crate::core_arch::simd::i64x4 {
        unsafe { transmute(self.as_m256i()) }
    }
}

impl m256iExt for __m256i {
    #[inline]
    fn as_m256i(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m128Ext: Sized {
    fn as_m128(self) -> __m128;

    #[inline]
    fn as_f32x4(self) -> crate::core_arch::simd::f32x4 {
        unsafe { transmute(self.as_m128()) }
    }
}

impl m128Ext for __m128 {
    #[inline]
    fn as_m128(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m128dExt: Sized {
    fn as_m128d(self) -> __m128d;

    #[inline]
    fn as_f64x2(self) -> crate::core_arch::simd::f64x2 {
        unsafe { transmute(self.as_m128d()) }
    }
}

impl m128dExt for __m128d {
    #[inline]
    fn as_m128d(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m256Ext: Sized {
    fn as_m256(self) -> __m256;

    #[inline]
    fn as_f32x8(self) -> crate::core_arch::simd::f32x8 {
        unsafe { transmute(self.as_m256()) }
    }
}

impl m256Ext for __m256 {
    #[inline]
    fn as_m256(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m512iExt: Sized {
    fn as_m512i(self) -> __m512i;

    #[inline]
    fn as_i8x64(self) -> crate::core_arch::simd::i8x64 {
        unsafe { transmute(self.as_m512i()) }
    }

    #[inline]
    fn as_i16x32(self) -> crate::core_arch::simd::i16x32 {
        unsafe { transmute(self.as_m512i()) }
    }

    #[inline]
    fn as_u32x16(self) -> crate::core_arch::simd::u32x16 {
        unsafe { transmute(self.as_m512i()) }
    }

    #[inline]
    fn as_i32x16(self) -> crate::core_arch::simd::i32x16 {
        unsafe { transmute(self.as_m512i()) }
    }

    #[inline]
    fn as_u64x8(self) -> crate::core_arch::simd::u64x8 {
        unsafe { transmute(self.as_m512i()) }
    }

    #[inline]
    fn as_i64x8(self) -> crate::core_arch::simd::i64x8 {
        unsafe { transmute(self.as_m512i()) }
    }
}

impl m512iExt for __m512i {
    #[inline]
    fn as_m512i(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m512Ext: Sized {
    fn as_m512(self) -> __m512;

    #[inline]
    fn as_f32x16(self) -> crate::core_arch::simd::f32x16 {
        unsafe { transmute(self.as_m512()) }
    }
}

impl m512Ext for __m512 {
    #[inline]
    fn as_m512(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[unstable(feature = "stdsimd_internal", issue = "none")]
pub(crate) trait m512dExt: Sized {
    fn as_m512d(self) -> __m512d;

    #[inline]
    fn as_f64x8(self) -> crate::core_arch::simd::f64x8 {
        unsafe { transmute(self.as_m512d()) }
    }
}

impl m512dExt for __m512d {
    #[inline]
    fn as_m512d(self) -> Self {
        self
    }
}

mod eflags;
pub use self::eflags::*;

mod fxsr;
pub use self::fxsr::*;

mod bswap;
pub use self::bswap::*;

mod rdtsc;
pub use self::rdtsc::*;

mod cpuid;
pub use self::cpuid::*;
mod xsave;
pub use self::xsave::*;

mod sse;
pub use self::sse::*;
mod sse2;
pub use self::sse2::*;
mod sse3;
pub use self::sse3::*;
mod ssse3;
pub use self::ssse3::*;
mod sse41;
pub use self::sse41::*;
mod sse42;
pub use self::sse42::*;
mod avx;
pub use self::avx::*;
mod avx2;
pub use self::avx2::*;
mod fma;
pub use self::fma::*;

mod abm;
pub use self::abm::*;
mod bmi1;
pub use self::bmi1::*;

mod bmi2;
pub use self::bmi2::*;

#[cfg(not(stdarch_intel_sde))]
mod sse4a;
#[cfg(not(stdarch_intel_sde))]
pub use self::sse4a::*;

#[cfg(not(stdarch_intel_sde))]
mod tbm;
#[cfg(not(stdarch_intel_sde))]
pub use self::tbm::*;

mod pclmulqdq;
pub use self::pclmulqdq::*;

mod aes;
pub use self::aes::*;

mod rdrand;
pub use self::rdrand::*;

mod sha;
pub use self::sha::*;

mod adx;
pub use self::adx::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `UD2`
#[cfg_attr(test, assert_instr(ud2))]
#[inline]
pub unsafe fn ud2() -> ! {
    intrinsics::abort()
}

mod avx512f;
pub use self::avx512f::*;

mod avx512ifma;
pub use self::avx512ifma::*;

mod bt;
pub use self::bt::*;

mod rtm;
pub use self::rtm::*;

mod f16c;
pub use self::f16c::*;

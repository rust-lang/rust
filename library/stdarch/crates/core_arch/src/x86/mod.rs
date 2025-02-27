//! `x86` and `x86_64` intrinsics.

use crate::mem::transmute;

#[macro_use]
mod macros;

types! {
    #![stable(feature = "simd_x86", since = "1.27.0")]

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
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
    /// # #[allow(unused_unsafe)] // temporary, to unstick CI
    /// # unsafe fn foo() { unsafe {
    /// let all_bytes_zero = _mm_setzero_si128();
    /// let all_bytes_one = _mm_set1_epi8(1);
    /// let four_i32 = _mm_set_epi32(1, 2, 3, 4);
    /// # }}
    /// # if is_x86_feature_detected!("sse2") { unsafe { foo() } }
    /// # }
    /// ```
    pub struct __m128i(2 x i64);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
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
    /// # #[allow(unused_unsafe)] // temporary, to unstick CI
    /// # unsafe fn foo() { unsafe {
    /// let four_zeros = _mm_setzero_ps();
    /// let four_ones = _mm_set1_ps(1.0);
    /// let four_floats = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    /// # }}
    /// # if is_x86_feature_detected!("sse") { unsafe { foo() } }
    /// # }
    /// ```
    pub struct __m128(4 x f32);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
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
    /// # #[target_feature(enable = "sse2")]
    /// # #[allow(unused_unsafe)] // temporary, to unstick CI
    /// # unsafe fn foo() { unsafe {
    /// let two_zeros = _mm_setzero_pd();
    /// let two_ones = _mm_set1_pd(1.0);
    /// let two_floats = _mm_set_pd(1.0, 2.0);
    /// # }}
    /// # if is_x86_feature_detected!("sse2") { unsafe { foo() } }
    /// # }
    /// ```
    pub struct __m128d(2 x f64);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
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
    /// # #[allow(unused_unsafe)] // temporary, to unstick CI
    /// # unsafe fn foo() { unsafe {
    /// let all_bytes_zero = _mm256_setzero_si256();
    /// let all_bytes_one = _mm256_set1_epi8(1);
    /// let eight_i32 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    /// # }}
    /// # if is_x86_feature_detected!("avx") { unsafe { foo() } }
    /// # }
    /// ```
    pub struct __m256i(4 x i64);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding  between two consecutive elements); however, the
    /// alignment is different and equal to the size of the type. Note that the
    /// ABI for function calls may *not* be the same.
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
    /// # #[allow(unused_unsafe)] // temporary, to unstick CI
    /// # unsafe fn foo() { unsafe {
    /// let eight_zeros = _mm256_setzero_ps();
    /// let eight_ones = _mm256_set1_ps(1.0);
    /// let eight_floats = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// # }}
    /// # if is_x86_feature_detected!("avx") { unsafe { foo() } }
    /// # }
    /// ```
    pub struct __m256(8 x f32);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
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
    /// # #[allow(unused_unsafe)] // temporary, to unstick CI
    /// # unsafe fn foo() { unsafe {
    /// let four_zeros = _mm256_setzero_pd();
    /// let four_ones = _mm256_set1_pd(1.0);
    /// let four_floats = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    /// # }}
    /// # if is_x86_feature_detected!("avx") { unsafe { foo() } }
    /// # }
    /// ```
    pub struct __m256d(4 x f64);
}

types! {
    #![stable(feature = "simd_avx512_types", since = "1.72.0")]

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    ///
    /// Note that this means that an instance of `__m512i` typically just means
    /// a "bag of bits" which is left up to interpretation at the point of use.
    pub struct __m512i(8 x i64);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding  between two consecutive elements); however, the
    /// alignment is different and equal to the size of the type. Note that the
    /// ABI for function calls may *not* be the same.
    ///
    /// Most intrinsics using `__m512` are prefixed with `_mm512_` and are
    /// suffixed with "ps" (or otherwise contain "ps"). Not to be confused with
    /// "pd" which is used for `__m512d`.
    pub struct __m512(16 x f32);

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
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding  between two consecutive elements); however, the
    /// alignment is different and equal to the size of the type. Note that the
    /// ABI for function calls may *not* be the same.
    ///
    /// Most intrinsics using `__m512d` are prefixed with `_mm512_` and are
    /// suffixed with "pd" (or otherwise contain "pd"). Not to be confused with
    /// "ps" which is used for `__m512`.
    pub struct __m512d(8 x f64);
}

types! {
    #![unstable(feature = "stdarch_x86_avx512", issue = "111137")]

    /// 128-bit wide set of eight `u16` types, x86-specific
    ///
    /// This type is representing a 128-bit SIMD register which internally is consisted of
    /// eight packed `u16` instances. Its purpose is for bf16 related intrinsic
    /// implementations.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    pub struct __m128bh(8 x u16);

    /// 256-bit wide set of 16 `u16` types, x86-specific
    ///
    /// This type is the same as the `__m256bh` type defined by Intel,
    /// representing a 256-bit SIMD register which internally is consisted of
    /// 16 packed `u16` instances. Its purpose is for bf16 related intrinsic
    /// implementations.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    pub struct __m256bh(16 x u16);

    /// 512-bit wide set of 32 `u16` types, x86-specific
    ///
    /// This type is the same as the `__m512bh` type defined by Intel,
    /// representing a 512-bit SIMD register which internally is consisted of
    /// 32 packed `u16` instances. Its purpose is for bf16 related intrinsic
    /// implementations.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    pub struct __m512bh(32 x u16);
}

types! {
    #![unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]

    /// 128-bit wide set of 8 `f16` types, x86-specific
    ///
    /// This type is the same as the `__m128h` type defined by Intel,
    /// representing a 128-bit SIMD register which internally is consisted of
    /// 8 packed `f16` instances. its purpose is for f16 related intrinsic
    /// implementations.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    pub struct __m128h(8 x f16);

    /// 256-bit wide set of 16 `f16` types, x86-specific
    ///
    /// This type is the same as the `__m256h` type defined by Intel,
    /// representing a 256-bit SIMD register which internally is consisted of
    /// 16 packed `f16` instances. its purpose is for f16 related intrinsic
    /// implementations.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    pub struct __m256h(16 x f16);

    /// 512-bit wide set of 32 `f16` types, x86-specific
    ///
    /// This type is the same as the `__m512h` type defined by Intel,
    /// representing a 512-bit SIMD register which internally is consisted of
    /// 32 packed `f16` instances. its purpose is for f16 related intrinsic
    /// implementations.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    pub struct __m512h(32 x f16);
}

/// The BFloat16 type used in AVX-512 intrinsics.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
pub struct bf16(u16);

impl bf16 {
    /// Raw transmutation from `u16`
    #[inline]
    #[must_use]
    #[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
    pub const fn from_bits(bits: u16) -> bf16 {
        bf16(bits)
    }

    /// Raw transmutation to `u16`
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

/// The `__mmask64` type used in AVX-512 intrinsics, a 64-bit integer
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type __mmask64 = u64;

/// The `__mmask32` type used in AVX-512 intrinsics, a 32-bit integer
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type __mmask32 = u32;

/// The `__mmask16` type used in AVX-512 intrinsics, a 16-bit integer
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type __mmask16 = u16;

/// The `__mmask8` type used in AVX-512 intrinsics, a 8-bit integer
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type __mmask8 = u8;

/// The `_MM_CMPINT_ENUM` type used to specify comparison operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type _MM_CMPINT_ENUM = i32;

/// The `MM_MANTISSA_NORM_ENUM` type used to specify mantissa normalized operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type _MM_MANTISSA_NORM_ENUM = i32;

/// The `MM_MANTISSA_SIGN_ENUM` type used to specify mantissa signed operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type _MM_MANTISSA_SIGN_ENUM = i32;

/// The `MM_PERM_ENUM` type used to specify shuffle operations in AVX-512 intrinsics.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub type _MM_PERM_ENUM = i32;

#[cfg(test)]
mod test;
#[cfg(test)]
pub use self::test::*;

macro_rules! as_transmute {
    ($from:ty => $($name:ident -> $to:ident),* $(,)?) => {
        impl $from {$(
            #[inline]
            pub(crate) fn $name(self) -> crate::core_arch::simd::$to {
                unsafe { transmute(self) }
            }
        )*}
    };
}

as_transmute!(__m128i =>
    as_u8x16 -> u8x16,
    as_u16x8 -> u16x8,
    as_u32x4 -> u32x4,
    as_u64x2 -> u64x2,
    as_i8x16 -> i8x16,
    as_i16x8 -> i16x8,
    as_i32x4 -> i32x4,
    as_i64x2 -> i64x2,
);
as_transmute!(__m256i =>
    as_u8x32 -> u8x32,
    as_u16x16 -> u16x16,
    as_u32x8 -> u32x8,
    as_u64x4 -> u64x4,
    as_i8x32 -> i8x32,
    as_i16x16 -> i16x16,
    as_i32x8 -> i32x8,
    as_i64x4 -> i64x4,
);
as_transmute!(__m512i =>
    as_u8x64 -> u8x64,
    as_u16x32 -> u16x32,
    as_u32x16 -> u32x16,
    as_u64x8 -> u64x8,
    as_i8x64 -> i8x64,
    as_i16x32 -> i16x32,
    as_i32x16 -> i32x16,
    as_i64x8 -> i64x8,
);

as_transmute!(__m128 => as_f32x4 -> f32x4);
as_transmute!(__m128d => as_f64x2 -> f64x2);
as_transmute!(__m256 => as_f32x8 -> f32x8);
as_transmute!(__m256d => as_f64x4 -> f64x4);
as_transmute!(__m512 => as_f32x16 -> f32x16);
as_transmute!(__m512d => as_f64x8 -> f64x8);

as_transmute!(__m128bh =>
    as_u16x8 -> u16x8,
    as_u32x4 -> u32x4,
    as_i16x8 -> i16x8,
    as_i32x4 -> i32x4,
);
as_transmute!(__m256bh =>
    as_u16x16 -> u16x16,
    as_u32x8 -> u32x8,
    as_i16x16 -> i16x16,
    as_i32x8 -> i32x8,
);
as_transmute!(__m512bh =>
    as_u16x32 -> u16x32,
    as_u32x16 -> u32x16,
    as_i16x32 -> i16x32,
    as_i32x16 -> i32x16,
);

as_transmute!(__m128h => as_f16x8 -> f16x8);
as_transmute!(__m256h => as_f16x16 -> f16x16);
as_transmute!(__m512h => as_f16x32 -> f16x32);

mod eflags;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::eflags::*;

mod fxsr;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::fxsr::*;

mod bswap;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bswap::*;

mod rdtsc;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::rdtsc::*;

mod cpuid;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::cpuid::*;
mod xsave;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::xsave::*;

mod sse;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse::*;
mod sse2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse2::*;
mod sse3;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse3::*;
mod ssse3;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::ssse3::*;
mod sse41;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse41::*;
mod sse42;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse42::*;
mod avx;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::avx::*;
mod avx2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::avx2::*;
mod fma;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::fma::*;

mod abm;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::abm::*;
mod bmi1;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bmi1::*;

mod bmi2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bmi2::*;

mod sse4a;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse4a::*;

mod tbm;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::tbm::*;

mod pclmulqdq;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::pclmulqdq::*;

mod aes;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::aes::*;

mod rdrand;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::rdrand::*;

mod sha;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sha::*;

mod adx;
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub use self::adx::*;

#[cfg(test)]
use stdarch_test::assert_instr;

mod avx512f;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512f::*;

mod avx512bw;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512bw::*;

mod avx512cd;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512cd::*;

mod avx512dq;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512dq::*;

mod avx512ifma;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512ifma::*;

mod avx512vbmi;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512vbmi::*;

mod avx512vbmi2;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512vbmi2::*;

mod avx512vnni;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512vnni::*;

mod avx512bitalg;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512bitalg::*;

mod gfni;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::gfni::*;

mod avx512vpopcntdq;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512vpopcntdq::*;

mod vaes;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::vaes::*;

mod vpclmulqdq;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::vpclmulqdq::*;

mod bt;
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub use self::bt::*;

mod rtm;
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub use self::rtm::*;

mod f16c;
#[stable(feature = "x86_f16c_intrinsics", since = "1.68.0")]
pub use self::f16c::*;

mod avx512bf16;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512bf16::*;

mod avxneconvert;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avxneconvert::*;

mod avx512fp16;
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub use self::avx512fp16::*;

mod kl;
#[unstable(feature = "keylocker_x86", issue = "134813")]
pub use self::kl::*;

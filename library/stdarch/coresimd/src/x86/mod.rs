//! `x86` and `x86_64` intrinsics.

use core::mem;

#[macro_use]
mod macros;

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m128(f32, f32, f32, f32);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m128d(f64, f64);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m256(f32, f32, f32, f32, f32, f32, f32, f32);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m256d(f64, f64, f64, f64);

pub use v128::__m128i;
pub use v256::__m256i;
pub use v64::__m64;

#[cfg(test)]
mod test;
#[cfg(test)]
pub use self::test::*;

#[doc(hidden)]
#[allow(non_camel_case_types)]
trait m128iExt: Sized {
    fn as_m128i(self) -> __m128i;

    #[inline(always)]
    fn as_u8x16(self) -> ::v128::u8x16 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_u16x8(self) -> ::v128::u16x8 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_u32x4(self) -> ::v128::u32x4 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_u64x2(self) -> ::v128::u64x2 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_i8x16(self) -> ::v128::i8x16 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_i16x8(self) -> ::v128::i16x8 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_i32x4(self) -> ::v128::i32x4 {
        unsafe { mem::transmute(self.as_m128i()) }
    }

    #[inline(always)]
    fn as_i64x2(self) -> ::v128::i64x2 {
        unsafe { mem::transmute(self.as_m128i()) }
    }
}

impl m128iExt for __m128i {
    #[inline(always)]
    fn as_m128i(self) -> __m128i { self }
}

#[doc(hidden)]
#[allow(non_camel_case_types)]
trait m256iExt: Sized {
    fn as_m256i(self) -> __m256i;

    #[inline(always)]
    fn as_u8x32(self) -> ::v256::u8x32 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_u16x16(self) -> ::v256::u16x16 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_u32x8(self) -> ::v256::u32x8 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_u64x4(self) -> ::v256::u64x4 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_i8x32(self) -> ::v256::i8x32 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_i16x16(self) -> ::v256::i16x16 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_i32x8(self) -> ::v256::i32x8 {
        unsafe { mem::transmute(self.as_m256i()) }
    }

    #[inline(always)]
    fn as_i64x4(self) -> ::v256::i64x4 {
        unsafe { mem::transmute(self.as_m256i()) }
    }
}

impl m256iExt for __m256i {
    #[inline(always)]
    fn as_m256i(self) -> __m256i { self }
}

mod i386;
pub use self::i386::*;

// x86 w/o sse2
mod i586;
pub use self::i586::*;

// `i686` is `i586 + sse2`.
//
// This module is not available for `i586` targets,
// but available for all `i686` targets by default
#[cfg(any(all(target_arch = "x86", target_feature = "sse2"),
          target_arch = "x86_64"))]
mod i686;
#[cfg(any(all(target_arch = "x86", target_feature = "sse2"),
          target_arch = "x86_64"))]
pub use self::i686::*;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use self::x86_64::*;

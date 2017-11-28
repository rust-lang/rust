//! `x86` and `x86_64` intrinsics.

#[macro_use]
mod macros;

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

/// 64-bit wide integer vector type.
#[allow(non_camel_case_types)]
#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct __m64(i64); // corresponds to llvm's `x86_mmx` type
/// 128-bit wide signed integer vector type
#[allow(non_camel_case_types)]
pub type __m128i = ::v128::i8x16;
/// 256-bit wide signed integer vector type
#[allow(non_camel_case_types)]
pub type __m256i = ::v256::i8x32;

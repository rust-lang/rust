//! `x86` and `x86_64` intrinsics.

pub use self::ia32::*;
pub use self::cpuid::*;
pub use self::xsave::*;

pub use self::sse::*;
pub use self::sse2::*;
pub use self::sse3::*;
pub use self::ssse3::*;
pub use self::sse41::*;
pub use self::sse42::*;
pub use self::avx::*;
pub use self::avx2::*;

pub use self::abm::*;
pub use self::bmi::*;
pub use self::bmi2::*;

#[cfg(not(feature = "intel_sde"))]
pub use self::tbm::*;

/// 128-bit wide signed integer vector type
#[allow(non_camel_case_types)]
pub type __m128i = ::v128::i8x16;
/// 256-bit wide signed integer vector type
#[allow(non_camel_case_types)]
pub type __m256i = ::v256::i8x32;

#[macro_use]
mod macros;

mod ia32;
mod cpuid;
mod xsave;

mod sse;
mod sse2;
mod sse3;
mod ssse3;
mod sse41;
mod sse42;
mod avx;
mod avx2;

mod abm;
mod bmi;
mod bmi2;

#[cfg(not(feature = "intel_sde"))]
mod tbm;

#[allow(non_camel_case_types)]
#[cfg(not(feature = "std"))]
#[repr(u8)]
pub enum c_void {
    #[doc(hidden)] __variant1,
    #[doc(hidden)] __variant2,
}

#[cfg(feature = "std")]
use std::os::raw::c_void;

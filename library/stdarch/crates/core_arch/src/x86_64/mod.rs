//! `x86_64` intrinsics

#[macro_use]
mod macros;

mod fxsr;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::fxsr::*;

mod sse;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse::*;

mod sse2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse2::*;

mod sse41;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse41::*;

mod sse42;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse42::*;

mod xsave;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::xsave::*;

mod abm;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::abm::*;

mod avx;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::avx::*;

mod bmi;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bmi::*;
mod bmi2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bmi2::*;

mod avx2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::avx2::*;

mod avx512f;
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub use self::avx512f::*;

mod bswap;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bswap::*;

mod rdrand;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::rdrand::*;

mod cmpxchg16b;
#[stable(feature = "cmpxchg16b_intrinsic", since = "1.67.0")]
pub use self::cmpxchg16b::*;

mod adx;
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub use self::adx::*;

mod bt;
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub use self::bt::*;

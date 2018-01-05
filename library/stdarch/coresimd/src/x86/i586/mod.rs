//! `i586` intrinsics

pub use self::cpuid::*;
pub use self::xsave::*;

pub use self::bswap::*;

pub use self::rdtsc::*;

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

mod cpuid;
mod xsave;

mod bswap;

mod rdtsc;

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

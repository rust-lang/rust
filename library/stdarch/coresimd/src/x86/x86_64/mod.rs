//! `x86_64` intrinsics

#[cfg(dont_compile_me)] // TODO: need to upstream `fxsr` target feature
mod fxsr;
#[cfg(dont_compile_me)] // TODO: need to upstream `fxsr` target feature
pub use self::fxsr::*;

mod sse;
pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

mod sse41;
pub use self::sse41::*;

mod sse42;
pub use self::sse42::*;

mod xsave;
pub use self::xsave::*;

mod abm;
pub use self::abm::*;

mod avx;
pub use self::avx::*;

mod bmi;
pub use self::bmi::*;

mod bmi2;
pub use self::bmi2::*;

mod avx2;
pub use self::avx2::*;

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

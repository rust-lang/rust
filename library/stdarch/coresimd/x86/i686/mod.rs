//! `i686` intrinsics

mod aes;
pub use self::aes::*;

mod mmx;
pub use self::mmx::*;

mod pclmulqdq;
pub use self::pclmulqdq::*;

mod sse;
pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

mod ssse3;
pub use self::ssse3::*;

mod sse41;
pub use self::sse41::*;

mod sse42;
pub use self::sse42::*;

#[cfg(not(feature = "intel_sde"))]
mod sse4a;
#[cfg(not(feature = "intel_sde"))]
pub use self::sse4a::*;

//! `i686` intrinsics

mod mmx;
pub use self::mmx::*;

mod sse;
pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

mod sse41;
pub use self::sse41::*;

mod sse42;
pub use self::sse42::*;

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
pub use self::tbm::*;

pub use self::runtime::{__unstable_detect_feature, __Feature};

#[allow(non_camel_case_types)]
pub type __m128i = ::v128::i8x16;
#[allow(non_camel_case_types)]
pub type __m256i = ::v256::i8x32;

#[macro_use]
mod macros;
#[macro_use]
mod runtime;

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
mod tbm;

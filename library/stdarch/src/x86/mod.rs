pub use self::sse::*;
pub use self::sse2::*;
pub use self::ssse3::*;
pub use self::sse41::*;
pub use self::sse42::*;
pub use self::avx::*;
pub use self::avx2::*;
pub use self::bmi::*;

#[allow(non_camel_case_types)]
pub type __m128i = ::v128::i8x16;
#[allow(non_camel_case_types)]
pub type __m256i = ::v256::i8x32;

mod sse;
mod sse2;
mod ssse3;
mod sse41;
mod sse42;
mod avx;
mod avx2;

mod bmi;
mod bmi2;

// pub use self::sse::*;
pub use self::sse2::*;
pub use self::ssse3::*;

#[allow(non_camel_case_types)]
pub type __m128i = ::v128::i8x16;

// mod sse;
mod sse2;
mod ssse3;

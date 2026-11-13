//! LoongArch64 LSX intrinsics

#![allow(non_camel_case_types)]

#[rustfmt::skip]
mod types;

#[rustfmt::skip]
#[stable(feature = "stdarch_loongarch_simd", since = "1.101.0")]
pub use self::types::*;

#[rustfmt::skip]
mod generated;

#[rustfmt::skip]
#[stable(feature = "stdarch_loongarch_simd", since = "1.101.0")]
pub use self::generated::*;

#[rustfmt::skip]
mod portable;

#[rustfmt::skip]
#[stable(feature = "stdarch_loongarch_simd", since = "1.101.0")]
pub use self::portable::*;

#[rustfmt::skip]
#[cfg(test)]
mod tests;

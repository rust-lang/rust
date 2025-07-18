//! LoongArch64 LSX intrinsics

#![allow(non_camel_case_types)]

#[rustfmt::skip]
mod types;

#[rustfmt::skip]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub use self::types::*;

#[rustfmt::skip]
mod generated;

#[rustfmt::skip]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub use self::generated::*;

#[rustfmt::skip]
#[cfg(test)]
mod tests;

//! `LoongArch` intrinsics

mod lsx;

#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub use self::lsx::*;

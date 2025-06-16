//! `s390x` intrinsics

pub(crate) mod macros;

mod vector;
#[unstable(feature = "stdarch_s390x", issue = "130869")]
pub use self::vector::*;

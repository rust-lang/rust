//! `s390x` intrinsics

pub(crate) mod macros;

/// the float and vector registers overlap therefore we cannot use any vector
/// extensions if softfloat is enabled.

#[cfg(not(target_abi = "softfloat"))]
mod vector;
#[cfg(not(target_abi = "softfloat"))]
#[unstable(feature = "stdarch_s390x", issue = "130869")]
pub use self::vector::*;

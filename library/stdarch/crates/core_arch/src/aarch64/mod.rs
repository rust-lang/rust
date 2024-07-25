//! AArch64 intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics

mod mte;
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub use self::mte::*;

// NEON intrinsics are currently broken on big-endian, so don't expose them. (#1484)
#[cfg(target_endian = "little")]
mod neon;
#[cfg(target_endian = "little")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub use self::neon::*;

mod tme;
#[unstable(feature = "stdarch_aarch64_tme", issue = "117216")]
pub use self::tme::*;

mod prefetch;
#[unstable(feature = "stdarch_aarch64_prefetch", issue = "117217")]
pub use self::prefetch::*;

#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub use super::arm_shared::*;

#[cfg(test)]
use stdarch_test::assert_instr;

#[cfg(test)]
pub(crate) mod test_support;

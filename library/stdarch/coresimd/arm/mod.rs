//! ARM intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]:
//! http://infocenter.arm.com/help/topic/com.arm.doc.
//! ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics

mod v6;
pub use self::v6::*;

mod v7;
pub use self::v7::*;

#[cfg(target_feature = "neon")]
mod neon;
#[cfg(target_feature = "neon")]
pub use self::neon::*;

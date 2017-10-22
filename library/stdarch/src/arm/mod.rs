//! ARM intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics

pub use self::v6::*;
pub use self::v7::*;
#[cfg(target_arch = "aarch64")]
pub use self::v8::*;

#[cfg(target_feature = "neon")]
pub use self::v7_neon::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use self::v8_neon::*;

mod v6;
mod v7;
#[cfg(target_feature = "neon")]
mod v7_neon;

#[cfg(target_arch = "aarch64")]
mod v8;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod v8_neon;

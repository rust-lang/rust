//! Architecture-specific support for Arm32/Thumb with hard float ABI.

mod sqrt;

#[cfg(target_feature = "vfp2")]
pub use sqrt::sqrt;
#[cfg(target_feature = "vfp2sp")]
pub use sqrt::sqrtf;

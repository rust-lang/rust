//! PowerPC intrinsics

#[cfg(target_feature = "altivec")]
mod altivec;
#[cfg(target_feature = "altivec")]
pub use self::altivec::*;

mod vsx;
pub use self::vsx::*;

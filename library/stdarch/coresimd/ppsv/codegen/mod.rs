//! Work arounds for code generation issues

#[cfg(target_arch = "aarch64")]
pub mod wrapping;

pub mod masks_reductions;

pub mod abs;
pub mod cos;
pub mod fma;
pub mod sin;
pub mod sqrt;

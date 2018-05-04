//! Work arounds for code generation issues

#[cfg(target_arch = "aarch64")]
pub mod wrapping;

pub mod masks_reductions;

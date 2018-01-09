//! Run-time feature detection for ARM and PowerPC64 on Linux.

#[cfg(target_arch = "arm")]
mod arm;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "powerpc64")]
mod powerpc64;

pub mod auxv;

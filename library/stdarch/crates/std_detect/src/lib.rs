//! Run-time feature detection for the Rust standard library.
//!
//! To detect whether a feature is enabled in the system running the binary
//! use one of the appropriate macro for the target:
//!
//! * `x86` and `x86_64`: [`is_x86_feature_detected`]
//! * `arm`: [`is_arm_feature_detected`]
//! * `aarch64`: [`is_aarch64_feature_detected`]
//! * `mips`: [`is_mips_feature_detected`]
//! * `mips64`: [`is_mips64_feature_detected`]
//! * `powerpc`: [`is_powerpc_feature_detected`]
//! * `powerpc64`: [`is_powerpc64_feature_detected`]

#![unstable(feature = "stdsimd", issue = "27731")]
#![feature(const_fn, staged_api, stdsimd, doc_cfg, allow_internal_unstable)]
#![allow(clippy::shadow_reuse)]
#![deny(clippy::missing_inline_in_public_items)]
#![cfg_attr(all(target_os = "freebsd", target_arch = "aarch64"), feature(asm))]
#![cfg_attr(test, allow(unused_imports))]
#![cfg_attr(feature = "std_detect_file_io", feature(vec_spare_capacity))]
#![no_std]

#[cfg_attr(feature = "rustc-dep-of-std", allow(unused_extern_crates))]
#[cfg(feature = "std_detect_file_io")]
extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

#[doc(hidden)]
#[unstable(feature = "stdsimd", issue = "27731")]
pub mod detect;

#![allow(dead_code)]
#![feature(
    const_fn, link_llvm_intrinsics, platform_intrinsics, repr_simd, simd_ffi,
    target_feature, cfg_target_feature, i128_type, asm, const_atomic_usize_new
)]
#![cfg_attr(test, feature(proc_macro))]

#[cfg(test)]
extern crate assert_instr;

/// Platform independent SIMD vector types and operations.
pub mod simd {
    pub use v128::*;
    pub use v256::*;
    pub use v512::*;
    pub use v64::*;
}

/// Platform dependent vendor intrinsics.
pub mod vendor {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use x86::*;

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    pub use arm::*;
}

#[macro_use]
mod macros;
mod simd_llvm;
mod v128;
mod v256;
mod v512;
mod v64;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
mod x86;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
 mod arm;

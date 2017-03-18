#![allow(dead_code)]
#![feature(
    const_fn, link_llvm_intrinsics, platform_intrinsics, repr_simd, simd_ffi,
    target_feature,
)]

pub use v128::*;
pub use v256::*;
pub use v64::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

#[macro_use]
mod macros;
mod simd;
mod v128;
mod v256;
mod v64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

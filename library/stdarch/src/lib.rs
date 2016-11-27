#![allow(dead_code)]
#![feature(link_llvm_intrinsics, platform_intrinsics, repr_simd, simd_ffi)]

// pub use v128::{__m128, __m128d, __m128i};
pub use v128::*;
pub use v64::__m64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

mod simd;
mod v128;
mod v64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

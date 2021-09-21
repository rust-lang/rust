#![cfg_attr(not(feature = "std"), no_std)]
#![allow(incomplete_features)]
#![feature(
    adt_const_params,
    const_fn_trait_bound,
    platform_intrinsics,
    repr_simd,
    simd_ffi,
    staged_api,
    stdsimd
)]
#![cfg_attr(feature = "generic_const_exprs", feature(generic_const_exprs))]
#![warn(missing_docs)]
#![unstable(feature = "portable_simd", issue = "86656")]
//! Portable SIMD module.

#[path = "mod.rs"]
mod core_simd;
use self::core_simd::simd;
pub use simd::*;

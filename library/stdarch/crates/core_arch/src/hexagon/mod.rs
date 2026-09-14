//! Hexagon architecture intrinsics
//!
//! This module contains intrinsics for the Qualcomm Hexagon DSP architecture,
//! including scalar operations and the Hexagon Vector Extensions (HVX).
//!
//! ## Scalar Intrinsics
//!
//! The [`scalar`] module provides intrinsics for scalar DSP operations including
//! arithmetic, multiply, shift, saturate, compare, and floating-point operations.
//!
//! ## HVX Vector Intrinsics
//!
//! HVX is a wide SIMD architecture designed for high-performance signal processing,
//! machine learning, and image processing workloads.
//!
//! HVX supports two vector length modes:
//! - 64-byte mode (512-bit vectors): Use the [`v64`] module
//! - 128-byte mode (1024-bit vectors): Use the [`v128`] module
//!
//! Both modules are available unconditionally, but require the appropriate
//! target features to actually use the intrinsics:
//! - For 64-byte mode: `-C target-feature=+hvx-length64b`
//! - For 128-byte mode: `-C target-feature=+hvx-length128b`
//!
//! Note that HVX v66 and later default to 128-byte mode, while earlier versions
//! (v60-v65) default to 64-byte mode.

/// Scalar intrinsics for Hexagon DSP operations
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub mod scalar;

/// HVX intrinsics for 64-byte vector mode (512-bit vectors)
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub mod v64;

/// HVX intrinsics for 128-byte vector mode (1024-bit vectors)
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub mod v128;

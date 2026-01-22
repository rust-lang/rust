//! Hexagon architecture intrinsics
//!
//! This module contains intrinsics for the Qualcomm Hexagon DSP architecture,
//! including the Hexagon Vector Extensions (HVX).
//!
//! HVX is a wide SIMD architecture designed for high-performance signal processing,
//! machine learning, and image processing workloads.

mod hvx;

pub use self::hvx::*;

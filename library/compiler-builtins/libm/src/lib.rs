//! libm in pure Rust
#![deny(warnings)]
#![no_std]
#![cfg_attr(
    all(target_arch = "wasm32", not(feature = "stable")),
    feature(core_intrinsics)
)]

mod math;

use core::{f32, f64};

pub use self::math::*;

/// Approximate equality with 1 ULP of tolerance
#[doc(hidden)]
#[inline]
pub fn _eqf(a: f32, b: f32) -> Result<(), u32> {
    if a.is_nan() && b.is_nan() {
        Ok(())
    } else {
        let err = (a.to_bits() as i32).wrapping_sub(b.to_bits() as i32).abs();

        if err <= 1 {
            Ok(())
        } else {
            Err(err as u32)
        }
    }
}

#[doc(hidden)]
#[inline]
pub fn _eq(a: f64, b: f64) -> Result<(), u64> {
    if a.is_nan() && b.is_nan() {
        Ok(())
    } else {
        let err = (a.to_bits() as i64).wrapping_sub(b.to_bits() as i64).abs();

        if err <= 1 {
            Ok(())
        } else {
            Err(err as u64)
        }
    }
}

#[cfg(all(test, feature = "musl-reference-tests"))]
include!(concat!(env!("OUT_DIR"), "/musl-tests.rs"));

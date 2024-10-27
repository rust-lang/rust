//! libm in pure Rust
#![no_std]
#![cfg_attr(feature = "unstable", allow(internal_features))]
#![cfg_attr(feature = "unstable", feature(core_intrinsics))]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::deprecated_cfg_attr)]
#![allow(clippy::eq_op)]
#![allow(clippy::float_cmp)]
#![allow(clippy::int_plus_one)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::mixed_case_hex_literals)]
#![allow(clippy::needless_return)]
#![allow(clippy::unreadable_literal)]

mod libm_helper;
mod math;

use core::{f32, f64};

pub use libm_helper::*;

pub use self::math::*;

/// Approximate equality with 1 ULP of tolerance
#[doc(hidden)]
#[inline]
pub fn _eqf(a: f32, b: f32) -> Result<(), u32> {
    if a.is_nan() && b.is_nan() {
        Ok(())
    } else {
        let err = (a.to_bits() as i32).wrapping_sub(b.to_bits() as i32).abs();

        if err <= 1 { Ok(()) } else { Err(err as u32) }
    }
}

#[doc(hidden)]
#[inline]
pub fn _eq(a: f64, b: f64) -> Result<(), u64> {
    if a.is_nan() && b.is_nan() {
        Ok(())
    } else {
        let err = (a.to_bits() as i64).wrapping_sub(b.to_bits() as i64).abs();

        if err <= 1 { Ok(()) } else { Err(err as u64) }
    }
}

//! libm in pure Rust
#![no_std]
#![cfg_attr(intrinsics_enabled, allow(internal_features))]
#![cfg_attr(intrinsics_enabled, feature(core_intrinsics))]
#![cfg_attr(f128_enabled, feature(f128))]
#![cfg_attr(f16_enabled, feature(f16))]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::deprecated_cfg_attr)]
#![allow(clippy::eq_op)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::float_cmp)]
#![allow(clippy::int_plus_one)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::mixed_case_hex_literals)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::needless_return)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::zero_divided_by_zero)]

mod libm_helper;
mod math;

use core::{f32, f64};

pub use libm_helper::*;

pub use self::math::*;

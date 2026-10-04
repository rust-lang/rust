//! libm in pure Rust
#![no_std]
//
#![cfg_attr(intrinsics_enabled, allow(internal_features))]
#![cfg_attr(intrinsics_enabled, feature(core_intrinsics))]
#![cfg_attr(
    all(intrinsics_enabled, target_family = "wasm"),
    feature(wasm_numeric_instr)
)]
#![cfg_attr(f128_enabled, feature(f128))]
#![cfg_attr(f16_enabled, feature(f16))]
//
// The edition is 2021 but we follow 2024 idioms.
#![deny(rust_2024_compatibility)]
#![allow(edition_2024_expr_fragment_specifier)]
//
// FIXME(float_bits_const): remove when stable
#![allow(unstable_name_collisions)]
// Allow idioms that come from ported C or may be more clear
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::int_plus_one)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::needless_return)]
// Literals are usually intentional
#![allow(clippy::excessive_precision)]
// Allow needed patterns like `(z - z) / (z - z)` and `0.0 / 0.0` that we need for exceptions
// and rounding.
#![allow(clippy::eq_op)]
#![allow(clippy::zero_divided_by_zero)]
// In generic code we use bits like `let _0 = F::ZERO`
#![allow(clippy::just_underscores_and_digits)]

mod libm_helper;
mod math;

use core::{f32, f64};

pub use libm_helper::*;

pub use self::math::*;

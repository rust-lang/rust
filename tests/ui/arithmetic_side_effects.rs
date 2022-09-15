#![allow(
    clippy::assign_op_pattern,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::unnecessary_owned_empty_strings,
    arithmetic_overflow,
    unconditional_panic
)]
#![feature(inline_const, saturating_int_impl)]
#![warn(clippy::arithmetic_side_effects)]

use core::num::{Saturating, Wrapping};

pub fn hard_coded_allowed() {
    let _ = 1f32 + 1f32;
    let _ = 1f64 + 1f64;

    let _ = Saturating(0u32) + Saturating(0u32);
    let _ = String::new() + "";
    let _ = Wrapping(0u32) + Wrapping(0u32);

    let saturating: Saturating<u32> = Saturating(0u32);
    let string: String = String::new();
    let wrapping: Wrapping<u32> = Wrapping(0u32);

    let inferred_saturating = saturating + saturating;
    let inferred_string = string + "";
    let inferred_wrapping = wrapping + wrapping;

    let _ = inferred_saturating + inferred_saturating;
    let _ = inferred_string + "";
    let _ = inferred_wrapping + inferred_wrapping;
}

#[rustfmt::skip]
pub fn non_overflowing_const_ops() {
    const _: i32 = { let mut n = 1; n += 1; n };
    let _ = const { let mut n = 1; n += 1; n };

    const _: i32 = { let mut n = 1; n = n + 1; n };
    let _ = const { let mut n = 1; n = n + 1; n };

    const _: i32 = { let mut n = 1; n = 1 + n; n };
    let _ = const { let mut n = 1; n = 1 + n; n };

    const _: i32 = 1 + 1;
    let _ = const { 1 + 1 };
}

pub fn non_overflowing_runtime_ops() {
    let mut _n = i32::MAX;

    // Assign
    _n += 0;
    _n -= 0;
    _n /= 99;
    _n %= 99;
    _n *= 0;
    _n *= 1;

    // Binary
    _n = _n + 0;
    _n = 0 + _n;
    _n = _n - 0;
    _n = 0 - _n;
    _n = _n / 99;
    _n = _n % 99;
    _n = _n * 0;
    _n = 0 * _n;
    _n = _n * 1;
    _n = 1 * _n;
    _n = 23 + 85;
}

#[rustfmt::skip]
pub fn overflowing_runtime_ops() {
    let mut _n = i32::MAX;

    // Assign
    _n += 1;
    _n -= 1;
    _n /= 0;
    _n %= 0;
    _n *= 2;

    // Binary
    _n = _n + 1;
    _n = 1 + _n;
    _n = _n - 1;
    _n = 1 - _n;
    _n = _n / 0;
    _n = _n % 0;
    _n = _n * 2;
    _n = 2 * _n;
}

fn main() {}

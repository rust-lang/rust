//! Note that we can use `rint` for these implementations since Rust expects the rounding
//! mode is always ties-to-even. `roundeven` also does not raise `FE_INEXACT`.
//!
//! Tested in the `rint` module.

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundevenf16(x: f16) -> f16 {
    super::rintf16(x)
}

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundevenf(x: f32) -> f32 {
    super::rintf(x)
}

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundeven(x: f64) -> f64 {
    super::rint(x)
}

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundevenf128(x: f128) -> f128 {
    super::rintf128(x)
}

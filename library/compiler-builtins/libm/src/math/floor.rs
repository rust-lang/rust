#![allow(unreachable_code)]
use core::f64;

const TOINT: f64 = 1. / f64::EPSILON;

/// Floor (f64)
///
/// Finds the nearest integer less than or equal to `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn floor(x: f64) -> f64 {
    select_implementation! {
        name: floor,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        use_arch_required: all(target_arch = "x86", not(target_feature = "sse2")),
        args: x,
    }

    let ui = x.to_bits();
    let e = ((ui >> 52) & 0x7ff) as i32;

    if (e >= 0x3ff + 52) || (x == 0.) {
        return x;
    }
    /* y = int(x) - x, where int(x) is an integer neighbor of x */
    let y = if (ui >> 63) != 0 { x - TOINT + TOINT - x } else { x + TOINT - TOINT - x };
    /* special case because of non-nearest rounding modes */
    if e < 0x3ff {
        force_eval!(y);
        return if (ui >> 63) != 0 { -1. } else { 0. };
    }
    if y > 0. { x + y - 1. } else { x + y }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_check() {
        assert_eq!(floor(1.1), 1.0);
        assert_eq!(floor(2.9), 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/floor
    #[test]
    fn spec_tests() {
        // Not Asserted: that the current rounding mode has no effect.
        assert!(floor(f64::NAN).is_nan());
        for f in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY].iter().copied() {
            assert_eq!(floor(f), f);
        }
    }
}

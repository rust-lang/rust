#![allow(unreachable_code)]
use core::f64;

const TOINT: f64 = 1. / f64::EPSILON;

/// Ceil (f64)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn ceil(x: f64) -> f64 {
    select_implementation! {
        name: ceil,
        use_intrinsic: target_arch = "wasm32",
        args: x,
    }

    #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
    {
        //use an alternative implementation on x86, because the
        //main implementation fails with the x87 FPU used by
        //debian i386, probably due to excess precision issues.
        //basic implementation taken from https://github.com/rust-lang/libm/issues/219
        use super::fabs;
        if fabs(x).to_bits() < 4503599627370496.0_f64.to_bits() {
            let truncated = x as i64 as f64;
            if truncated < x {
                return truncated + 1.0;
            } else {
                return truncated;
            }
        } else {
            return x;
        }
    }
    let u: u64 = x.to_bits();
    let e: i64 = (u >> 52 & 0x7ff) as i64;
    let y: f64;

    if e >= 0x3ff + 52 || x == 0. {
        return x;
    }
    // y = int(x) - x, where int(x) is an integer neighbor of x
    y = if (u >> 63) != 0 { x - TOINT + TOINT - x } else { x + TOINT - TOINT - x };
    // special case because of non-nearest rounding modes
    if e < 0x3ff {
        force_eval!(y);
        return if (u >> 63) != 0 { -0. } else { 1. };
    }
    if y < 0. { x + y + 1. } else { x + y }
}

#[cfg(test)]
mod tests {
    use core::f64::*;

    use super::*;

    #[test]
    fn sanity_check() {
        assert_eq!(ceil(1.1), 2.0);
        assert_eq!(ceil(2.9), 3.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/ceil
    #[test]
    fn spec_tests() {
        // Not Asserted: that the current rounding mode has no effect.
        assert!(ceil(NAN).is_nan());
        for f in [0.0, -0.0, INFINITY, NEG_INFINITY].iter().copied() {
            assert_eq!(ceil(f), f);
        }
    }
}

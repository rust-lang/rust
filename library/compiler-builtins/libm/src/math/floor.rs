#![allow(unreachable_code)]
use core::f64;

const TOINT: f64 = 1. / f64::EPSILON;

/// Floor (f64)
///
/// Finds the nearest integer less than or equal to `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn floor(x: f64) -> f64 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f64.floor` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32")] {
            return unsafe { ::core::intrinsics::floorf64(x) }
        }
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
            if truncated > x {
                return truncated - 1.0;
            } else {
                return truncated;
            }
        } else {
            return x;
        }
    }
    let ui = x.to_bits();
    let e = ((ui >> 52) & 0x7ff) as i32;

    if (e >= 0x3ff + 52) || (x == 0.) {
        return x;
    }
    /* y = int(x) - x, where int(x) is an integer neighbor of x */
    let y = if (ui >> 63) != 0 {
        x - TOINT + TOINT - x
    } else {
        x + TOINT - TOINT - x
    };
    /* special case because of non-nearest rounding modes */
    if e < 0x3ff {
        force_eval!(y);
        return if (ui >> 63) != 0 { -1. } else { 0. };
    }
    if y > 0. {
        x + y - 1.
    } else {
        x + y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::*;

    #[test]
    fn sanity_check() {
        assert_eq!(floor(1.1), 1.0);
        assert_eq!(floor(2.9), 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/floor
    #[test]
    fn spec_tests() {
        // Not Asserted: that the current rounding mode has no effect.
        assert!(floor(NAN).is_nan());
        for f in [0.0, -0.0, INFINITY, NEG_INFINITY].iter().copied() {
            assert_eq!(floor(f), f);
        }
    }
}

#![allow(unused_macros)]

/// Error tolerance in ULP.
/// Miri adds 4 ULP of errors itself, and we allow for 4 additional ULP of host float error.
pub const ERR_TOLERANCE: i32 = 8;

/// Compare the two floats, allowing for $ulp many ULPs of error.
///
/// ULP means "Units in the Last Place" or "Units of Least Precision".
/// The ULP of a float `a` is the smallest possible change at `a`, so the ULP difference represents how
/// many discrete floating-point steps are needed to reach the actual value from the expected value.
///
/// Essentially ULP can be seen as a distance metric of floating-point numbers, but with
/// the same amount of "spacing" between all consecutive representable values. So even though 2 very large floating point numbers
/// have a large value difference, their ULP can still be 1, so they are still "approximatly equal",
/// but the EPSILON check would have failed.
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $ulp:expr $( , )?) => {{
        let (actual, expected) = ($a, $b);
        let allowed_ulp_diff = $ulp;
        let _force_same_type = actual == expected;
        // Approximate the ULP by taking half the distance between the number one place "up"
        // and the number one place "down".
        let ulp = (expected.next_up() - expected.next_down()) / 2.0;
        let ulp_diff = ((actual - expected) / ulp).abs().round() as i32;
        if ulp_diff > allowed_ulp_diff {
            panic!("{actual:?} is not approximately equal to {expected:?}\ndifference in ULP: {ulp_diff} > {allowed_ulp_diff}");
        };
    }};

    ($a:expr, $b: expr $( , )?) => {
        // Accept up to 12ULP (4ULP for miri artificial error and the rest for host floats).
        // We saw failures on an i686-linux host with a limit of 8!
        $crate::utils::assert_approx_eq!($a, $b, $crate::utils::ERR_TOLERANCE);
    };
}

pub(crate) use assert_approx_eq;

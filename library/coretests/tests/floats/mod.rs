use std::fmt;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// Verify that floats are within a tolerance of each other, 1.0e-6 by default.
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{ assert_approx_eq!($a, $b, 1.0e-6) }};
    ($a:expr, $b:expr, $lim:expr) => {{
        let (a, b) = (&$a, &$b);
        let diff = (*a - *b).abs();
        assert!(
            diff <= $lim,
            "{a:?} is not approximately equal to {b:?} (threshold {lim:?}, difference {diff:?})",
            lim = $lim
        );
    }};
}

/// Verify that floats have the same bitwise representation. Used to avoid the default `0.0 == -0.0`
/// behavior, as well as to ensure exact NaN bitpatterns.
macro_rules! assert_biteq {
    (@inner $left:expr, $right:expr, $msg_sep:literal, $($tt:tt)*) => {{
        let l = $left;
        let r = $right;

        // Hack to coerce left and right to the same type
        let mut _eq_ty = l;
        _eq_ty = r;

        // Hack to get the width from a value
        let bits = (l.to_bits() - l.to_bits()).leading_zeros();
        assert!(
            l.to_bits() == r.to_bits(),
            "{msg}{nl}l: {l:?} ({lb:#0width$x})\nr: {r:?} ({rb:#0width$x})",
            msg = format_args!($($tt)*),
            nl = $msg_sep,
            lb = l.to_bits(),
            rb = r.to_bits(),
            width = ((bits / 4) + 2) as usize,
        );

        if !l.is_nan() && !r.is_nan() {
            // Also check that standard equality holds, since most tests use `assert_biteq` rather
            // than `assert_eq`.
            assert_eq!(l, r)
        }
    }};
    ($left:expr, $right:expr , $($tt:tt)*) => {
        assert_biteq!(@inner $left, $right, "\n", $($tt)*)
    };
    ($left:expr, $right:expr $(,)?) => {
        assert_biteq!(@inner $left, $right, "", "")
    };
}

/// Helper function for testing numeric operations
pub fn test_num<T>(ten: T, two: T)
where
    T: PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Rem<Output = T>
        + fmt::Debug
        + Copy,
{
    assert_eq!(ten.add(two), ten + two);
    assert_eq!(ten.sub(two), ten - two);
    assert_eq!(ten.mul(two), ten * two);
    assert_eq!(ten.div(two), ten / two);
    assert_eq!(ten.rem(two), ten % two);
}

mod f128;
mod f16;
mod f32;
mod f64;

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

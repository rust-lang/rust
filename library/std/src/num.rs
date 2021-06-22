//! Additional functionality for numerics.
//!
//! This module provides some extra types that are useful when doing numerical
//! work. See the individual documentation for each piece for more information.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

#[cfg(test)]
mod tests;

#[cfg(test)]
mod benches;

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::num::Wrapping;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::num::{FpCategory, ParseFloatError, ParseIntError, TryFromIntError};

#[stable(feature = "signed_nonzero", since = "1.34.0")]
pub use core::num::{NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize};
#[stable(feature = "nonzero", since = "1.28.0")]
pub use core::num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};

#[stable(feature = "int_error_matching", since = "1.55.0")]
pub use core::num::IntErrorKind;

#[cfg(test)]
use crate::fmt;
#[cfg(test)]
use crate::ops::{Add, Div, Mul, Rem, Sub};

/// Helper function for testing numeric operations
#[cfg(test)]
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

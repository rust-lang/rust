//! Additional functionality for numerics.
//!
//! This module provides some extra types that are useful when doing numerical
//! work. See the individual documentation for each piece for more information.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

#[cfg(test)]
mod tests;

#[stable(feature = "int_error_matching", since = "1.55.0")]
pub use core::num::IntErrorKind;
#[stable(feature = "generic_nonzero", since = "1.79.0")]
pub use core::num::NonZero;
#[stable(feature = "saturating_int_impl", since = "1.74.0")]
pub use core::num::Saturating;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::num::Wrapping;
#[unstable(
    feature = "nonzero_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
pub use core::num::ZeroablePrimitive;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::num::{FpCategory, ParseFloatError, ParseIntError, TryFromIntError};
#[stable(feature = "signed_nonzero", since = "1.34.0")]
pub use core::num::{NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize};
#[stable(feature = "nonzero", since = "1.28.0")]
pub use core::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize};

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

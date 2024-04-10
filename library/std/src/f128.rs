//! Constants for the `f128` double-precision floating point type.
//!
//! *[See also the `f128` primitive type](primitive@f128).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#[cfg(test)]
mod tests;

#[unstable(feature = "f128", issue = "116909")]
pub use core::f128::consts;

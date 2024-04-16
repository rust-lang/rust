//! Constants for the `f16` double-precision floating point type.
//!
//! *[See also the `f16` primitive type](primitive@f16).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#[cfg(test)]
mod tests;

#[unstable(feature = "f16", issue = "116909")]
pub use core::f16::consts;

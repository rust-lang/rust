//! Constants for the `f16` double-precision floating point type.
//!
//! *[See also the `f16` primitive type](primitive@f16).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#[cfg(test)]
mod tests;

#[cfg(not(test))]
use crate::intrinsics;

#[unstable(feature = "f16", issue = "116909")]
pub use core::f16::consts;

#[cfg(not(test))]
impl f16 {
    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`.
    /// It might have a different sequence of rounding operations than `powf`,
    /// so the results are not guaranteed to agree.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn powi(self, n: i32) -> f16 {
        unsafe { intrinsics::powif16(self, n) }
    }
}

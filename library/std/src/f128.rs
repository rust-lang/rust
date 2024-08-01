//! Constants for the `f128` double-precision floating point type.
//!
//! *[See also the `f128` primitive type](primitive@f128).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#[cfg(test)]
mod tests;

#[unstable(feature = "f128", issue = "116909")]
pub use core::f128::consts;

#[cfg(not(test))]
use crate::intrinsics;

#[cfg(not(test))]
impl f128 {
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
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn powi(self, n: i32) -> f128 {
        unsafe { intrinsics::powif128(self, n) }
    }

    /// Computes the absolute value of `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128)] { // FIXME(f16_f128): reliable_f128
    ///
    /// let x = 3.5_f128;
    /// let y = -3.5_f128;
    ///
    /// assert_eq!(x.abs(), x);
    /// assert_eq!(y.abs(), -y);
    ///
    /// assert!(f128::NAN.abs().is_nan());
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn abs(self) -> Self {
        // FIXME(f16_f128): replace with `intrinsics::fabsf128` when available
        // We don't do this now because LLVM has lowering bugs for f128 math.
        Self::from_bits(self.to_bits() & !(1 << 127))
    }
}

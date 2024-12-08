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
use crate::sys::cmath;

#[cfg(not(test))]
impl f128 {
    /// Returns the largest integer less than or equal to `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 3.7_f128;
    /// let g = 3.0_f128;
    /// let h = -3.7_f128;
    ///
    /// assert_eq!(f.floor(), 3.0);
    /// assert_eq!(g.floor(), 3.0);
    /// assert_eq!(h.floor(), -4.0);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn floor(self) -> f128 {
        unsafe { intrinsics::floorf128(self) }
    }

    /// Returns the smallest integer greater than or equal to `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 3.01_f128;
    /// let g = 4.0_f128;
    ///
    /// assert_eq!(f.ceil(), 4.0);
    /// assert_eq!(g.ceil(), 4.0);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "ceiling")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn ceil(self) -> f128 {
        unsafe { intrinsics::ceilf128(self) }
    }

    /// Returns the nearest integer to `self`. If a value is half-way between two
    /// integers, round away from `0.0`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 3.3_f128;
    /// let g = -3.3_f128;
    /// let h = -3.7_f128;
    /// let i = 3.5_f128;
    /// let j = 4.5_f128;
    ///
    /// assert_eq!(f.round(), 3.0);
    /// assert_eq!(g.round(), -3.0);
    /// assert_eq!(h.round(), -4.0);
    /// assert_eq!(i.round(), 4.0);
    /// assert_eq!(j.round(), 5.0);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn round(self) -> f128 {
        unsafe { intrinsics::roundf128(self) }
    }

    /// Returns the nearest integer to a number. Rounds half-way cases to the number
    /// with an even least significant digit.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 3.3_f128;
    /// let g = -3.3_f128;
    /// let h = 3.5_f128;
    /// let i = 4.5_f128;
    ///
    /// assert_eq!(f.round_ties_even(), 3.0);
    /// assert_eq!(g.round_ties_even(), -3.0);
    /// assert_eq!(h.round_ties_even(), 4.0);
    /// assert_eq!(i.round_ties_even(), 4.0);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn round_ties_even(self) -> f128 {
        unsafe { intrinsics::rintf128(self) }
    }

    /// Returns the integer part of `self`.
    /// This means that non-integer numbers are always truncated towards zero.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 3.7_f128;
    /// let g = 3.0_f128;
    /// let h = -3.7_f128;
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), 3.0);
    /// assert_eq!(h.trunc(), -3.0);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "truncate")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn trunc(self) -> f128 {
        unsafe { intrinsics::truncf128(self) }
    }

    /// Returns the fractional part of `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 3.6_f128;
    /// let y = -3.6_f128;
    /// let abs_difference_x = (x.fract() - 0.6).abs();
    /// let abs_difference_y = (y.fract() - (-0.6)).abs();
    ///
    /// assert!(abs_difference_x <= f128::EPSILON);
    /// assert!(abs_difference_y <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn fract(self) -> f128 {
        self - self.trunc()
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if
    /// the target architecture has a dedicated `fma` CPU instruction. However,
    /// this is not always true, and will be heavily dependant on designing
    /// algorithms with specific target hardware in mind.
    ///
    /// # Precision
    ///
    /// The result of this operation is guaranteed to be the rounded
    /// infinite-precision result. It is specified by IEEE 754 as
    /// `fusedMultiplyAdd` and guaranteed not to change.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let m = 10.0_f128;
    /// let x = 4.0_f128;
    /// let b = 60.0_f128;
    ///
    /// assert_eq!(m.mul_add(x, b), 100.0);
    /// assert_eq!(m * x + b, 100.0);
    ///
    /// let one_plus_eps = 1.0_f128 + f128::EPSILON;
    /// let one_minus_eps = 1.0_f128 - f128::EPSILON;
    /// let minus_one = -1.0_f128;
    ///
    /// // The exact result (1 + eps) * (1 - eps) = 1 - eps * eps.
    /// assert_eq!(one_plus_eps.mul_add(one_minus_eps, minus_one), -f128::EPSILON * f128::EPSILON);
    /// // Different rounding with the non-fused multiply and add.
    /// assert_eq!(one_plus_eps * one_minus_eps + minus_one, 0.0);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn mul_add(self, a: f128, b: f128) -> f128 {
        unsafe { intrinsics::fmaf128(self, a, b) }
    }

    /// Calculates Euclidean division, the matching method for `rem_euclid`.
    ///
    /// This computes the integer `n` such that
    /// `self = n * rhs + self.rem_euclid(rhs)`.
    /// In other words, the result is `self / rhs` rounded to the integer `n`
    /// such that `self >= n * rhs`.
    ///
    /// # Precision
    ///
    /// The result of this operation is guaranteed to be the rounded
    /// infinite-precision result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let a: f128 = 7.0;
    /// let b = 4.0;
    /// assert_eq!(a.div_euclid(b), 1.0); // 7.0 > 4.0 * 1.0
    /// assert_eq!((-a).div_euclid(b), -2.0); // -7.0 >= 4.0 * -2.0
    /// assert_eq!(a.div_euclid(-b), -1.0); // 7.0 >= -4.0 * -1.0
    /// assert_eq!((-a).div_euclid(-b), 2.0); // -7.0 >= -4.0 * 2.0
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn div_euclid(self, rhs: f128) -> f128 {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
    /// most cases. However, due to a floating point round-off error it can
    /// result in `r == rhs.abs()`, violating the mathematical definition, if
    /// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
    /// This result is not an element of the function's codomain, but it is the
    /// closest floating point number in the real numbers and thus fulfills the
    /// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
    /// approximately.
    ///
    /// # Precision
    ///
    /// The result of this operation is guaranteed to be the rounded
    /// infinite-precision result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let a: f128 = 7.0;
    /// let b = 4.0;
    /// assert_eq!(a.rem_euclid(b), 3.0);
    /// assert_eq!((-a).rem_euclid(b), 1.0);
    /// assert_eq!(a.rem_euclid(-b), 3.0);
    /// assert_eq!((-a).rem_euclid(-b), 1.0);
    /// // limitation due to round-off error
    /// assert!((-f128::EPSILON).rem_euclid(3.0) != 0.0);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[doc(alias = "modulo", alias = "mod")]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn rem_euclid(self, rhs: f128) -> f128 {
        let r = self % rhs;
        if r < 0.0 { r + rhs.abs() } else { r }
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`.
    /// It might have a different sequence of rounding operations than `powf`,
    /// so the results are not guaranteed to agree.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn powi(self, n: i32) -> f128 {
        unsafe { intrinsics::powif128(self, n) }
    }

    /// Raises a number to a floating point power.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 2.0_f128;
    /// let abs_difference = (x.powf(2.0) - (x * x)).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn powf(self, n: f128) -> f128 {
        unsafe { intrinsics::powf128(self, n) }
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    ///
    /// # Precision
    ///
    /// The result of this operation is guaranteed to be the rounded
    /// infinite-precision result. It is specified by IEEE 754 as `squareRoot`
    /// and guaranteed not to change.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let positive = 4.0_f128;
    /// let negative = -4.0_f128;
    /// let negative_zero = -0.0_f128;
    ///
    /// assert_eq!(positive.sqrt(), 2.0);
    /// assert!(negative.sqrt().is_nan());
    /// assert!(negative_zero.sqrt() == negative_zero);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn sqrt(self) -> f128 {
        unsafe { intrinsics::sqrtf128(self) }
    }

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let one = 1.0f128;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn exp(self) -> f128 {
        unsafe { intrinsics::expf128(self) }
    }

    /// Returns `2^(self)`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 2.0f128;
    ///
    /// // 2^2 - 4 == 0
    /// let abs_difference = (f.exp2() - 4.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn exp2(self) -> f128 {
        unsafe { intrinsics::exp2f128(self) }
    }

    /// Returns the natural logarithm of the number.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let one = 1.0f128;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn ln(self) -> f128 {
        unsafe { intrinsics::logf128(self) }
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let five = 5.0f128;
    ///
    /// // log5(5) - 1 == 0
    /// let abs_difference = (five.log(5.0) - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn log(self, base: f128) -> f128 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let two = 2.0f128;
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference = (two.log2() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn log2(self) -> f128 {
        unsafe { intrinsics::log2f128(self) }
    }

    /// Returns the base 10 logarithm of the number.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let ten = 10.0f128;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference = (ten.log10() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn log10(self) -> f128 {
        unsafe { intrinsics::log10f128(self) }
    }

    /// Returns the cube root of a number.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    ///
    /// This function currently corresponds to the `cbrtf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 8.0f128;
    ///
    /// // x^(1/3) - 2 == 0
    /// let abs_difference = (x.cbrt() - 2.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn cbrt(self) -> f128 {
        unsafe { cmath::cbrtf128(self) }
    }

    /// Compute the distance between the origin and a point (`x`, `y`) on the
    /// Euclidean plane. Equivalently, compute the length of the hypotenuse of a
    /// right-angle triangle with other sides having length `x.abs()` and
    /// `y.abs()`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    ///
    /// This function currently corresponds to the `hypotf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 2.0f128;
    /// let y = 3.0f128;
    ///
    /// // sqrt(x^2 + y^2)
    /// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn hypot(self, other: f128) -> f128 {
        unsafe { cmath::hypotf128(self, other) }
    }

    /// Computes the sine of a number (in radians).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = std::f128::consts::FRAC_PI_2;
    ///
    /// let abs_difference = (x.sin() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn sin(self) -> f128 {
        unsafe { intrinsics::sinf128(self) }
    }

    /// Computes the cosine of a number (in radians).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 2.0 * std::f128::consts::PI;
    ///
    /// let abs_difference = (x.cos() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn cos(self) -> f128 {
        unsafe { intrinsics::cosf128(self) }
    }

    /// Computes the tangent of a number (in radians).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `tanf128` from libc on Unix and
    /// Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = std::f128::consts::FRAC_PI_4;
    /// let abs_difference = (x.tan() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn tan(self) -> f128 {
        unsafe { cmath::tanf128(self) }
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `asinf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = std::f128::consts::FRAC_PI_2;
    ///
    /// // asin(sin(pi/2))
    /// let abs_difference = (f.sin().asin() - std::f128::consts::FRAC_PI_2).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "arcsin")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn asin(self) -> f128 {
        unsafe { cmath::asinf128(self) }
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `acosf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = std::f128::consts::FRAC_PI_4;
    ///
    /// // acos(cos(pi/4))
    /// let abs_difference = (f.cos().acos() - std::f128::consts::FRAC_PI_4).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "arccos")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn acos(self) -> f128 {
        unsafe { cmath::acosf128(self) }
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `atanf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let f = 1.0f128;
    ///
    /// // atan(tan(1))
    /// let abs_difference = (f.tan().atan() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "arctan")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn atan(self) -> f128 {
        unsafe { cmath::atanf128(self) }
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `atan2f128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// // Positive angles measured counter-clockwise
    /// // from positive x axis
    /// // -pi/4 radians (45 deg clockwise)
    /// let x1 = 3.0f128;
    /// let y1 = -3.0f128;
    ///
    /// // 3pi/4 radians (135 deg counter-clockwise)
    /// let x2 = -3.0f128;
    /// let y2 = 3.0f128;
    ///
    /// let abs_difference_1 = (y1.atan2(x1) - (-std::f128::consts::FRAC_PI_4)).abs();
    /// let abs_difference_2 = (y2.atan2(x2) - (3.0 * std::f128::consts::FRAC_PI_4)).abs();
    ///
    /// assert!(abs_difference_1 <= f128::EPSILON);
    /// assert!(abs_difference_2 <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn atan2(self, other: f128) -> f128 {
        unsafe { cmath::atan2f128(self, other) }
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `(f128::sin(x),
    /// f128::cos(x))`. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = std::f128::consts::FRAC_PI_4;
    /// let f = x.sin_cos();
    ///
    /// let abs_difference_0 = (f.0 - x.sin()).abs();
    /// let abs_difference_1 = (f.1 - x.cos()).abs();
    ///
    /// assert!(abs_difference_0 <= f128::EPSILON);
    /// assert!(abs_difference_1 <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "sincos")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn sin_cos(self) -> (f128, f128) {
        (self.sin(), self.cos())
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `expm1f128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 1e-8_f128;
    ///
    /// // for very small x, e^x is approximately 1 + x + x^2 / 2
    /// let approx = x + x * x / 2.0;
    /// let abs_difference = (x.exp_m1() - approx).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn exp_m1(self) -> f128 {
        unsafe { cmath::expm1f128(self) }
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `log1pf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 1e-8_f128;
    ///
    /// // for very small x, ln(1 + x) is approximately x - x^2 / 2
    /// let approx = x - x * x / 2.0;
    /// let abs_difference = (x.ln_1p() - approx).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "log1p")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn ln_1p(self) -> f128 {
        unsafe { cmath::log1pf128(self) }
    }

    /// Hyperbolic sine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `sinhf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let e = std::f128::consts::E;
    /// let x = 1.0f128;
    ///
    /// let f = x.sinh();
    /// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
    /// let g = ((e * e) - 1.0) / (2.0 * e);
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn sinh(self) -> f128 {
        unsafe { cmath::sinhf128(self) }
    }

    /// Hyperbolic cosine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `coshf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let e = std::f128::consts::E;
    /// let x = 1.0f128;
    /// let f = x.cosh();
    /// // Solving cosh() at 1 gives this result
    /// let g = ((e * e) + 1.0) / (2.0 * e);
    /// let abs_difference = (f - g).abs();
    ///
    /// // Same result
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn cosh(self) -> f128 {
        unsafe { cmath::coshf128(self) }
    }

    /// Hyperbolic tangent function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `tanhf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let e = std::f128::consts::E;
    /// let x = 1.0f128;
    ///
    /// let f = x.tanh();
    /// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
    /// let g = (1.0 - e.powi(-2)) / (1.0 + e.powi(-2));
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn tanh(self) -> f128 {
        unsafe { cmath::tanhf128(self) }
    }

    /// Inverse hyperbolic sine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 1.0f128;
    /// let f = x.sinh().asinh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "arcsinh")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn asinh(self) -> f128 {
        let ax = self.abs();
        let ix = 1.0 / ax;
        (ax + (ax / (Self::hypot(1.0, ix) + ix))).ln_1p().copysign(self)
    }

    /// Inverse hyperbolic cosine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 1.0f128;
    /// let f = x.cosh().acosh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "arccosh")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn acosh(self) -> f128 {
        if self < 1.0 {
            Self::NAN
        } else {
            (self + ((self - 1.0).sqrt() * (self + 1.0).sqrt())).ln()
        }
    }

    /// Inverse hyperbolic tangent function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let e = std::f128::consts::E;
    /// let f = e.tanh().atanh();
    ///
    /// let abs_difference = (f - e).abs();
    ///
    /// assert!(abs_difference <= 1e-5);
    /// # }
    /// ```
    #[inline]
    #[doc(alias = "arctanh")]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn atanh(self) -> f128 {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }

    /// Gamma function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `tgammaf128` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// #![feature(float_gamma)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 5.0f128;
    ///
    /// let abs_difference = (x.gamma() - 24.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn gamma(self) -> f128 {
        unsafe { cmath::tgammaf128(self) }
    }

    /// Natural logarithm of the absolute value of the gamma function
    ///
    /// The integer part of the tuple indicates the sign of the gamma function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `lgammaf128_r` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// #![feature(float_gamma)]
    /// # #[cfg(reliable_f128_math)] {
    ///
    /// let x = 2.0f128;
    ///
    /// let abs_difference = (x.ln_gamma().0 - 0.0).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn ln_gamma(self) -> (f128, i32) {
        let mut signgamp: i32 = 0;
        let x = unsafe { cmath::lgammaf128_r(self, &mut signgamp) };
        (x, signgamp)
    }
}

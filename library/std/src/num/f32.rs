//! Constants for the `f32` single-precision floating point type.
//!
//! *[See also the `f32` primitive type](primitive@f32).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.
//!
//! For the constants defined directly in this module
//! (as distinct from those defined in the `consts` sub-module),
//! new code should instead use the associated constants
//! defined directly on the `f32` type.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::f32::{
    DIGITS, EPSILON, INFINITY, MANTISSA_DIGITS, MAX, MAX_10_EXP, MAX_EXP, MIN, MIN_10_EXP, MIN_EXP,
    MIN_POSITIVE, NAN, NEG_INFINITY, RADIX, consts,
};

#[cfg(not(test))]
use crate::intrinsics;
#[cfg(not(test))]
use crate::sys::cmath;

#[cfg(not(test))]
impl f32 {
    /// Returns the largest integer less than or equal to `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.7_f32;
    /// let g = 3.0_f32;
    /// let h = -3.7_f32;
    ///
    /// assert_eq!(f.floor(), 3.0);
    /// assert_eq!(g.floor(), 3.0);
    /// assert_eq!(h.floor(), -4.0);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_round_methods", since = "1.90.0")]
    #[inline]
    pub const fn floor(self) -> f32 {
        core::f32::math::floor(self)
    }

    /// Returns the smallest integer greater than or equal to `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.01_f32;
    /// let g = 4.0_f32;
    ///
    /// assert_eq!(f.ceil(), 4.0);
    /// assert_eq!(g.ceil(), 4.0);
    /// ```
    #[doc(alias = "ceiling")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_round_methods", since = "1.90.0")]
    #[inline]
    pub const fn ceil(self) -> f32 {
        core::f32::math::ceil(self)
    }

    /// Returns the nearest integer to `self`. If a value is half-way between two
    /// integers, round away from `0.0`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.3_f32;
    /// let g = -3.3_f32;
    /// let h = -3.7_f32;
    /// let i = 3.5_f32;
    /// let j = 4.5_f32;
    ///
    /// assert_eq!(f.round(), 3.0);
    /// assert_eq!(g.round(), -3.0);
    /// assert_eq!(h.round(), -4.0);
    /// assert_eq!(i.round(), 4.0);
    /// assert_eq!(j.round(), 5.0);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_round_methods", since = "1.90.0")]
    #[inline]
    pub const fn round(self) -> f32 {
        core::f32::math::round(self)
    }

    /// Returns the nearest integer to a number. Rounds half-way cases to the number
    /// with an even least significant digit.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.3_f32;
    /// let g = -3.3_f32;
    /// let h = 3.5_f32;
    /// let i = 4.5_f32;
    ///
    /// assert_eq!(f.round_ties_even(), 3.0);
    /// assert_eq!(g.round_ties_even(), -3.0);
    /// assert_eq!(h.round_ties_even(), 4.0);
    /// assert_eq!(i.round_ties_even(), 4.0);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "round_ties_even", since = "1.77.0")]
    #[rustc_const_stable(feature = "const_float_round_methods", since = "1.90.0")]
    #[inline]
    pub const fn round_ties_even(self) -> f32 {
        core::f32::math::round_ties_even(self)
    }

    /// Returns the integer part of `self`.
    /// This means that non-integer numbers are always truncated towards zero.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.7_f32;
    /// let g = 3.0_f32;
    /// let h = -3.7_f32;
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), 3.0);
    /// assert_eq!(h.trunc(), -3.0);
    /// ```
    #[doc(alias = "truncate")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_round_methods", since = "1.90.0")]
    #[inline]
    pub const fn trunc(self) -> f32 {
        core::f32::math::trunc(self)
    }

    /// Returns the fractional part of `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 3.6_f32;
    /// let y = -3.6_f32;
    /// let abs_difference_x = (x.fract() - 0.6).abs();
    /// let abs_difference_y = (y.fract() - (-0.6)).abs();
    ///
    /// assert!(abs_difference_x <= f32::EPSILON);
    /// assert!(abs_difference_y <= f32::EPSILON);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_round_methods", since = "1.90.0")]
    #[inline]
    pub const fn fract(self) -> f32 {
        core::f32::math::fract(self)
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
    /// let m = 10.0_f32;
    /// let x = 4.0_f32;
    /// let b = 60.0_f32;
    ///
    /// assert_eq!(m.mul_add(x, b), 100.0);
    /// assert_eq!(m * x + b, 100.0);
    ///
    /// let one_plus_eps = 1.0_f32 + f32::EPSILON;
    /// let one_minus_eps = 1.0_f32 - f32::EPSILON;
    /// let minus_one = -1.0_f32;
    ///
    /// // The exact result (1 + eps) * (1 - eps) = 1 - eps * eps.
    /// assert_eq!(one_plus_eps.mul_add(one_minus_eps, minus_one), -f32::EPSILON * f32::EPSILON);
    /// // Different rounding with the non-fused multiply and add.
    /// assert_eq!(one_plus_eps * one_minus_eps + minus_one, 0.0);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[doc(alias = "fmaf", alias = "fusedMultiplyAdd")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[rustc_const_unstable(feature = "const_mul_add", issue = "146724")]
    pub const fn mul_add(self, a: f32, b: f32) -> f32 {
        core::f32::math::mul_add(self, a, b)
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
    /// let a: f32 = 7.0;
    /// let b = 4.0;
    /// assert_eq!(a.div_euclid(b), 1.0); // 7.0 > 4.0 * 1.0
    /// assert_eq!((-a).div_euclid(b), -2.0); // -7.0 >= 4.0 * -2.0
    /// assert_eq!(a.div_euclid(-b), -1.0); // 7.0 >= -4.0 * -1.0
    /// assert_eq!((-a).div_euclid(-b), 2.0); // -7.0 >= -4.0 * 2.0
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    #[stable(feature = "euclidean_division", since = "1.38.0")]
    pub fn div_euclid(self, rhs: f32) -> f32 {
        core::f32::math::div_euclid(self, rhs)
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
    /// let a: f32 = 7.0;
    /// let b = 4.0;
    /// assert_eq!(a.rem_euclid(b), 3.0);
    /// assert_eq!((-a).rem_euclid(b), 1.0);
    /// assert_eq!(a.rem_euclid(-b), 3.0);
    /// assert_eq!((-a).rem_euclid(-b), 1.0);
    /// // limitation due to round-off error
    /// assert!((-f32::EPSILON).rem_euclid(3.0) != 0.0);
    /// ```
    #[doc(alias = "modulo", alias = "mod")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    #[stable(feature = "euclidean_division", since = "1.38.0")]
    pub fn rem_euclid(self, rhs: f32) -> f32 {
        core::f32::math::rem_euclid(self, rhs)
    }

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
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0_f32;
    /// let abs_difference = (x.powi(2) - (x * x)).abs();
    /// assert!(abs_difference <= 1e-5);
    ///
    /// assert_eq!(f32::powi(f32::NAN, 0), 1.0);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn powi(self, n: i32) -> f32 {
        core::f32::math::powi(self, n)
    }

    /// Raises a number to a floating point power.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0_f32;
    /// let abs_difference = (x.powf(2.0) - (x * x)).abs();
    /// assert!(abs_difference <= 1e-5);
    ///
    /// assert_eq!(f32::powf(1.0, f32::NAN), 1.0);
    /// assert_eq!(f32::powf(f32::NAN, 0.0), 1.0);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn powf(self, n: f32) -> f32 {
        intrinsics::powf32(self, n)
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
    /// let positive = 4.0_f32;
    /// let negative = -4.0_f32;
    /// let negative_zero = -0.0_f32;
    ///
    /// assert_eq!(positive.sqrt(), 2.0);
    /// assert!(negative.sqrt().is_nan());
    /// assert!(negative_zero.sqrt() == negative_zero);
    /// ```
    #[doc(alias = "squareRoot")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sqrt(self) -> f32 {
        core::f32::math::sqrt(self)
    }

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let one = 1.0f32;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn exp(self) -> f32 {
        intrinsics::expf32(self)
    }

    /// Returns `2^(self)`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 2.0f32;
    ///
    /// // 2^2 - 4 == 0
    /// let abs_difference = (f.exp2() - 4.0).abs();
    ///
    /// assert!(abs_difference <= 1e-5);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn exp2(self) -> f32 {
        intrinsics::exp2f32(self)
    }

    /// Returns the natural logarithm of the number.
    ///
    /// This returns NaN when the number is negative, and negative infinity when number is zero.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let one = 1.0f32;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    ///
    /// Non-positive values:
    /// ```
    /// assert_eq!(0_f32.ln(), f32::NEG_INFINITY);
    /// assert!((-42_f32).ln().is_nan());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn ln(self) -> f32 {
        intrinsics::logf32(self)
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// This returns NaN when the number is negative, and negative infinity when number is zero.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = 5.0f32;
    ///
    /// // log5(5) - 1 == 0
    /// let abs_difference = (five.log(5.0) - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    ///
    /// Non-positive values:
    /// ```
    /// assert_eq!(0_f32.log(10.0), f32::NEG_INFINITY);
    /// assert!((-42_f32).log(10.0).is_nan());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn log(self, base: f32) -> f32 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    ///
    /// This returns NaN when the number is negative, and negative infinity when number is zero.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let two = 2.0f32;
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference = (two.log2() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    ///
    /// Non-positive values:
    /// ```
    /// assert_eq!(0_f32.log2(), f32::NEG_INFINITY);
    /// assert!((-42_f32).log2().is_nan());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn log2(self) -> f32 {
        intrinsics::log2f32(self)
    }

    /// Returns the base 10 logarithm of the number.
    ///
    /// This returns NaN when the number is negative, and negative infinity when number is zero.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let ten = 10.0f32;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference = (ten.log10() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    ///
    /// Non-positive values:
    /// ```
    /// assert_eq!(0_f32.log10(), f32::NEG_INFINITY);
    /// assert!((-42_f32).log10().is_nan());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn log10(self) -> f32 {
        intrinsics::log10f32(self)
    }

    /// The positive difference of two numbers.
    ///
    /// * If `self <= other`: `0.0`
    /// * Else: `self - other`
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `fdimf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 3.0f32;
    /// let y = -3.0f32;
    ///
    /// let abs_difference_x = (x.abs_sub(1.0) - 2.0).abs();
    /// let abs_difference_y = (y.abs_sub(1.0) - 0.0).abs();
    ///
    /// assert!(abs_difference_x <= 1e-6);
    /// assert!(abs_difference_y <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[deprecated(
        since = "1.10.0",
        note = "you probably meant `(self - other).abs()`: \
                this operation is `(self - other).max(0.0)` \
                except that `abs_sub` also propagates NaNs (also \
                known as `fdimf` in C). If you truly need the positive \
                difference, consider using that expression or the C function \
                `fdimf`, depending on how you wish to handle NaN (please consider \
                filing an issue describing your use-case too)."
    )]
    pub fn abs_sub(self, other: f32) -> f32 {
        #[allow(deprecated)]
        core::f32::math::abs_sub(self, other)
    }

    /// Returns the cube root of a number.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `cbrtf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 8.0f32;
    ///
    /// // x^(1/3) - 2 == 0
    /// let abs_difference = (x.cbrt() - 2.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn cbrt(self) -> f32 {
        core::f32::math::cbrt(self)
    }

    /// Compute the distance between the origin and a point (`x`, `y`) on the
    /// Euclidean plane. Equivalently, compute the length of the hypotenuse of a
    /// right-angle triangle with other sides having length `x.abs()` and
    /// `y.abs()`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `hypotf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0f32;
    /// let y = 3.0f32;
    ///
    /// // sqrt(x^2 + y^2)
    /// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
    ///
    /// assert!(abs_difference <= 1e-5);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn hypot(self, other: f32) -> f32 {
        cmath::hypotf(self, other)
    }

    /// Computes the sine of a number (in radians).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = std::f32::consts::FRAC_PI_2;
    ///
    /// let abs_difference = (x.sin() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sin(self) -> f32 {
        intrinsics::sinf32(self)
    }

    /// Computes the cosine of a number (in radians).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0 * std::f32::consts::PI;
    ///
    /// let abs_difference = (x.cos() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn cos(self) -> f32 {
        intrinsics::cosf32(self)
    }

    /// Computes the tangent of a number (in radians).
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `tanf` from libc on Unix and
    /// Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = std::f32::consts::FRAC_PI_4;
    /// let abs_difference = (x.tan() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn tan(self) -> f32 {
        cmath::tanf(self)
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `asinf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = std::f32::consts::FRAC_PI_4;
    ///
    /// // asin(sin(pi/2))
    /// let abs_difference = (f.sin().asin() - f).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[doc(alias = "arcsin")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn asin(self) -> f32 {
        cmath::asinf(self)
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `acosf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = std::f32::consts::FRAC_PI_4;
    ///
    /// // acos(cos(pi/4))
    /// let abs_difference = (f.cos().acos() - std::f32::consts::FRAC_PI_4).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[doc(alias = "arccos")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn acos(self) -> f32 {
        cmath::acosf(self)
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `atanf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 1.0f32;
    ///
    /// // atan(tan(1))
    /// let abs_difference = (f.tan().atan() - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[doc(alias = "arctan")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn atan(self) -> f32 {
        cmath::atanf(self)
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    ///
    ///  | `x`     | `y`     | Piecewise Definition | Range         |
    ///  |---------|---------|----------------------|---------------|
    ///  | `>= +0` | `>= +0` | `arctan(y/x)`        | `[+0, +pi/2]` |
    ///  | `>= +0` | `<= -0` | `arctan(y/x)`        | `[-pi/2, -0]` |
    ///  | `<= -0` | `>= +0` | `arctan(y/x) + pi`   | `[+pi/2, +pi]`|
    ///  | `<= -0` | `<= -0` | `arctan(y/x) - pi`   | `[-pi, -pi/2]`|
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `atan2f` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// // Positive angles measured counter-clockwise
    /// // from positive x axis
    /// // -pi/4 radians (45 deg clockwise)
    /// let x1 = 3.0f32;
    /// let y1 = -3.0f32;
    ///
    /// // 3pi/4 radians (135 deg counter-clockwise)
    /// let x2 = -3.0f32;
    /// let y2 = 3.0f32;
    ///
    /// let abs_difference_1 = (y1.atan2(x1) - (-std::f32::consts::FRAC_PI_4)).abs();
    /// let abs_difference_2 = (y2.atan2(x2) - (3.0 * std::f32::consts::FRAC_PI_4)).abs();
    ///
    /// assert!(abs_difference_1 <= 1e-5);
    /// assert!(abs_difference_2 <= 1e-5);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn atan2(self, other: f32) -> f32 {
        cmath::atan2f(self, other)
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `(f32::sin(x),
    /// f32::cos(x))`. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = std::f32::consts::FRAC_PI_4;
    /// let f = x.sin_cos();
    ///
    /// let abs_difference_0 = (f.0 - x.sin()).abs();
    /// let abs_difference_1 = (f.1 - x.cos()).abs();
    ///
    /// assert!(abs_difference_0 <= 1e-4);
    /// assert!(abs_difference_1 <= 1e-4);
    /// ```
    #[doc(alias = "sincos")]
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sin_cos(self) -> (f32, f32) {
        (self.sin(), self.cos())
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `expm1f` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 1e-8_f32;
    ///
    /// // for very small x, e^x is approximately 1 + x + x^2 / 2
    /// let approx = x + x * x / 2.0;
    /// let abs_difference = (x.exp_m1() - approx).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn exp_m1(self) -> f32 {
        cmath::expm1f(self)
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    ///
    /// This returns NaN when `n < -1.0`, and negative infinity when `n == -1.0`.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `log1pf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 1e-8_f32;
    ///
    /// // for very small x, ln(1 + x) is approximately x - x^2 / 2
    /// let approx = x - x * x / 2.0;
    /// let abs_difference = (x.ln_1p() - approx).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    ///
    /// Out-of-range values:
    /// ```
    /// assert_eq!((-1.0_f32).ln_1p(), f32::NEG_INFINITY);
    /// assert!((-2.0_f32).ln_1p().is_nan());
    /// ```
    #[doc(alias = "log1p")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn ln_1p(self) -> f32 {
        cmath::log1pf(self)
    }

    /// Hyperbolic sine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `sinhf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let e = std::f32::consts::E;
    /// let x = 1.0f32;
    ///
    /// let f = x.sinh();
    /// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
    /// let g = ((e * e) - 1.0) / (2.0 * e);
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sinh(self) -> f32 {
        cmath::sinhf(self)
    }

    /// Hyperbolic cosine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `coshf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let e = std::f32::consts::E;
    /// let x = 1.0f32;
    /// let f = x.cosh();
    /// // Solving cosh() at 1 gives this result
    /// let g = ((e * e) + 1.0) / (2.0 * e);
    /// let abs_difference = (f - g).abs();
    ///
    /// // Same result
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn cosh(self) -> f32 {
        cmath::coshf(self)
    }

    /// Hyperbolic tangent function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `tanhf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// let e = std::f32::consts::E;
    /// let x = 1.0f32;
    ///
    /// let f = x.tanh();
    /// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
    /// let g = (1.0 - e.powi(-2)) / (1.0 + e.powi(-2));
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn tanh(self) -> f32 {
        cmath::tanhf(self)
    }

    /// Inverse hyperbolic sine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 1.0f32;
    /// let f = x.sinh().asinh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[doc(alias = "arcsinh")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn asinh(self) -> f32 {
        let ax = self.abs();
        let ix = 1.0 / ax;
        (ax + (ax / (Self::hypot(1.0, ix) + ix))).ln_1p().copysign(self)
    }

    /// Inverse hyperbolic cosine function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 1.0f32;
    /// let f = x.cosh().acosh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[doc(alias = "arccosh")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn acosh(self) -> f32 {
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
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = std::f32::consts::FRAC_PI_6;
    /// let f = x.tanh().atanh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference <= 1e-5);
    /// ```
    #[doc(alias = "arctanh")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn atanh(self) -> f32 {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }

    /// Gamma function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `tgammaf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(float_gamma)]
    /// let x = 5.0f32;
    ///
    /// let abs_difference = (x.gamma() - 24.0).abs();
    ///
    /// assert!(abs_difference <= 1e-5);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[unstable(feature = "float_gamma", issue = "99842")]
    #[inline]
    pub fn gamma(self) -> f32 {
        cmath::tgammaf(self)
    }

    /// Natural logarithm of the absolute value of the gamma function
    ///
    /// The integer part of the tuple indicates the sign of the gamma function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
    /// can even differ within the same execution from one invocation to the next.
    /// This function currently corresponds to the `lgamma_r` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(float_gamma)]
    /// let x = 2.0f32;
    ///
    /// let abs_difference = (x.ln_gamma().0 - 0.0).abs();
    ///
    /// assert!(abs_difference <= f32::EPSILON);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[unstable(feature = "float_gamma", issue = "99842")]
    #[inline]
    pub fn ln_gamma(self) -> (f32, i32) {
        let mut signgamp: i32 = 0;
        let x = cmath::lgammaf_r(self, &mut signgamp);
        (x, signgamp)
    }

    /// Error function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `erff` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(float_erf)]
    /// /// The error function relates what percent of a normal distribution lies
    /// /// within `x` standard deviations (scaled by `1/sqrt(2)`).
    /// fn within_standard_deviations(x: f32) -> f32 {
    ///     (x * std::f32::consts::FRAC_1_SQRT_2).erf() * 100.0
    /// }
    ///
    /// // 68% of a normal distribution is within one standard deviation
    /// assert!((within_standard_deviations(1.0) - 68.269).abs() < 0.01);
    /// // 95% of a normal distribution is within two standard deviations
    /// assert!((within_standard_deviations(2.0) - 95.450).abs() < 0.01);
    /// // 99.7% of a normal distribution is within three standard deviations
    /// assert!((within_standard_deviations(3.0) - 99.730).abs() < 0.01);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[unstable(feature = "float_erf", issue = "136321")]
    #[inline]
    pub fn erf(self) -> f32 {
        cmath::erff(self)
    }

    /// Complementary error function.
    ///
    /// # Unspecified precision
    ///
    /// The precision of this function is non-deterministic. This means it varies by platform,
    /// Rust version, and can even differ within the same execution from one invocation to the next.
    ///
    /// This function currently corresponds to the `erfcf` from libc on Unix
    /// and Windows. Note that this might change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(float_erf)]
    /// let x: f32 = 0.123;
    ///
    /// let one = x.erf() + x.erfc();
    /// let abs_difference = (one - 1.0).abs();
    ///
    /// assert!(abs_difference <= 1e-6);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[unstable(feature = "float_erf", issue = "136321")]
    #[inline]
    pub fn erfc(self) -> f32 {
        cmath::erfcf(self)
    }
}

//! Constants for the `f16` double-precision floating point type.
//!
//! *[See also the `f16` primitive type](primitive@f16).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#[cfg(test)]
#[path = "tests/f16_tests.rs"]
mod tests;

#[cfg(not(test))]
use crate::intrinsics;
// extended math currently disabled
// #[cfg(not(test))]
// use crate::sys::cmath;

#[unstable(feature = "f16", issue = "116909")]
pub use core::f16::consts;

#[cfg(not(test))]
impl f16 {
    /// Returns the largest integer less than or equal to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.7_f16;
    /// let g = 3.0_f16;
    /// let h = -3.7_f16;
    ///
    /// assert_eq!(f.floor(), 3.0);
    /// assert_eq!(g.floor(), 3.0);
    /// assert_eq!(h.floor(), -4.0);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[unstable(feature = "f16_math", issue = "116909")]
    pub fn floor(self) -> f16 {
        unsafe { intrinsics::floorf16(self) }
    }

    /// Returns the smallest integer greater than or equal to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.01_f16;
    /// let g = 4.0_f16;
    ///
    /// assert_eq!(f.ceil(), 4.0);
    /// assert_eq!(g.ceil(), 4.0);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn ceil(self) -> f16 {
        unsafe { intrinsics::ceilf16(self) }
    }

    /// Returns the nearest integer to `self`. If a value is half-way between two
    /// integers, round away from `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.3_f16;
    /// let g = -3.3_f16;
    /// let h = -3.7_f16;
    /// let i = 3.5_f16;
    /// let j = 4.5_f16;
    ///
    /// assert_eq!(f.round(), 3.0);
    /// assert_eq!(g.round(), -3.0);
    /// assert_eq!(h.round(), -4.0);
    /// assert_eq!(i.round(), 4.0);
    /// assert_eq!(j.round(), 5.0);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn round(self) -> f16 {
        unsafe { intrinsics::roundf16(self) }
    }

    /// Returns the nearest integer to a number. Rounds half-way cases to the number
    /// with an even least significant digit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(round_ties_even)]
    ///
    /// let f = 3.3_f16;
    /// let g = -3.3_f16;
    /// let h = 3.5_f16;
    /// let i = 4.5_f16;
    ///
    /// assert_eq!(f.round_ties_even(), 3.0);
    /// assert_eq!(g.round_ties_even(), -3.0);
    /// assert_eq!(h.round_ties_even(), 4.0);
    /// assert_eq!(i.round_ties_even(), 4.0);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "round_ties_even", issue = "96710")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn round_ties_even(self) -> f16 {
        unsafe { intrinsics::rintf16(self) }
    }

    /// Returns the integer part of `self`.
    /// This means that non-integer numbers are always truncated towards zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.7_f16;
    /// let g = 3.0_f16;
    /// let h = -3.7_f16;
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), 3.0);
    /// assert_eq!(h.trunc(), -3.0);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn trunc(self) -> f16 {
        unsafe { intrinsics::truncf16(self) }
    }

    /// Returns the fractional part of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 3.6_f16;
    /// let y = -3.6_f16;
    /// let abs_difference_x = (x.fract() - 0.6).abs();
    /// let abs_difference_y = (y.fract() - (-0.6)).abs();
    ///
    /// assert!(abs_difference_x <= f16::EPSILON);
    /// assert!(abs_difference_y <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn fract(self) -> f16 {
        self - self.trunc()
    }

    /// Computes the absolute value of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 3.5_f16;
    /// let y = -3.5_f16;
    ///
    /// let abs_difference_x = (x.abs() - x).abs();
    /// let abs_difference_y = (y.abs() - (-y)).abs();
    ///
    /// assert!(abs_difference_x <= f16::EPSILON);
    /// assert!(abs_difference_y <= f16::EPSILON);
    ///
    /// assert!(f16::NAN.abs().is_nan());
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn abs(self) -> f16 {
        unsafe { intrinsics::fabsf16(self) }
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - NaN if the number is NaN
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.5_f16;
    ///
    /// assert_eq!(f.signum(), 1.0);
    /// assert_eq!(f16::NEG_INFINITY.signum(), -1.0);
    ///
    /// assert!(f16::NAN.signum().is_nan());
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn signum(self) -> f16 {
        if self.is_nan() { Self::NAN } else { 1.0_f16.copysign(self) }
    }

    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    ///
    /// Equal to `self` if the sign of `self` and `sign` are the same, otherwise
    /// equal to `-self`. If `self` is a NaN, then a NaN with the sign bit of
    /// `sign` is returned. Note, however, that conserving the sign bit on NaN
    /// across arithmetical operations is not generally guaranteed.
    /// See [explanation of NaN as a special value](primitive@f16) for more info.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.5_f16;
    ///
    /// assert_eq!(f.copysign(0.42), 3.5_f16);
    /// assert_eq!(f.copysign(-0.42), -3.5_f16);
    /// assert_eq!((-f).copysign(0.42), 3.5_f16);
    /// assert_eq!((-f).copysign(-0.42), -3.5_f16);
    ///
    /// assert!(f16::NAN.copysign(1.0).is_nan());
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn copysign(self, sign: f16) -> f16 {
        unsafe { intrinsics::copysignf16(self, sign) }
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if
    /// the target architecture has a dedicated `fma` CPU instruction. However,
    /// this is not always true, and will be heavily dependant on designing
    /// algorithms with specific target hardware in mind.
    ///
    /// # Examples
    ///
    /// ```
    /// let m = 10.0_f16;
    /// let x = 4.0_f16;
    /// let b = 60.0_f16;
    ///
    /// // 100.0
    /// let abs_difference = (m.mul_add(x, b) - ((m * x) + b)).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn mul_add(self, a: f16, b: f16) -> f16 {
        unsafe { intrinsics::fmaf16(self, a, b) }
    }

    /// Calculates Euclidean division, the matching method for `rem_euclid`.
    ///
    /// This computes the integer `n` such that
    /// `self = n * rhs + self.rem_euclid(rhs)`.
    /// In other words, the result is `self / rhs` rounded to the integer `n`
    /// such that `self >= n * rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: f16 = 7.0;
    /// let b = 4.0;
    /// assert_eq!(a.div_euclid(b), 1.0); // 7.0 > 4.0 * 1.0
    /// assert_eq!((-a).div_euclid(b), -2.0); // -7.0 >= 4.0 * -2.0
    /// assert_eq!(a.div_euclid(-b), -1.0); // 7.0 >= -4.0 * -1.0
    /// assert_eq!((-a).div_euclid(-b), 2.0); // -7.0 >= -4.0 * 2.0
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn div_euclid(self, rhs: f16) -> f16 {
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
    /// # Examples
    ///
    /// ```
    /// let a: f16 = 7.0;
    /// let b = 4.0;
    /// assert_eq!(a.rem_euclid(b), 3.0);
    /// assert_eq!((-a).rem_euclid(b), 1.0);
    /// assert_eq!(a.rem_euclid(-b), 3.0);
    /// assert_eq!((-a).rem_euclid(-b), 1.0);
    /// // limitation due to round-off error
    /// assert!((-f16::EPSILON).rem_euclid(3.0) != 0.0);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn rem_euclid(self, rhs: f16) -> f16 {
        let r = self % rhs;
        if r < 0.0 { r + rhs.abs() } else { r }
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`.
    /// It might have a different sequence of rounding operations than `powf`,
    /// so the results are not guaranteed to agree.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0_f16;
    /// let abs_difference = (x.powi(2) - (x * x)).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn powi(self, n: i32) -> f16 {
        unsafe { intrinsics::powif16(self, n) }
    }

    /// Raises a number to a floating point power.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0_f16;
    /// let abs_difference = (x.powf(2.0) - (x * x)).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn powf(self, n: f16) -> f16 {
        unsafe { intrinsics::powf16(self, n) }
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// let positive = 4.0_f16;
    /// let negative = -4.0_f16;
    /// let negative_zero = -0.0_f16;
    ///
    /// let abs_difference = (positive.sqrt() - 2.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// assert!(negative.sqrt().is_nan());
    /// assert!(negative_zero.sqrt() == negative_zero);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn sqrt(self) -> f16 {
        unsafe { intrinsics::sqrtf16(self) }
    }

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// # Examples
    ///
    /// ```
    /// let one = 1.0f16;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn exp(self) -> f16 {
        unsafe { intrinsics::expf16(self) }
    }

    /// Returns `2^(self)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 2.0f16;
    ///
    /// // 2^2 - 4 == 0
    /// let abs_difference = (f.exp2() - 4.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn exp2(self) -> f16 {
        unsafe { intrinsics::exp2f16(self) }
    }

    /// Returns the natural logarithm of the number.
    ///
    /// # Examples
    ///
    /// ```
    /// let one = 1.0f16;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn ln(self) -> f16 {
        unsafe { intrinsics::logf16(self) }
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2, and
    /// `self.log10()` can produce more accurate results for base 10.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = 5.0f16;
    ///
    /// // log5(5) - 1 == 0
    /// let abs_difference = (five.log(5.0) - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn log(self, base: f16) -> f16 {
        self.ln() / base.ln()
    }

    /// Returns the base 2 logarithm of the number.
    ///
    /// # Examples
    ///
    /// ```
    /// let two = 2.0f16;
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference = (two.log2() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn log2(self) -> f16 {
        crate::sys::log2f16(self)
    }

    /// Returns the base 10 logarithm of the number.
    ///
    /// # Examples
    ///
    /// ```
    /// let ten = 10.0f16;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference = (ten.log10() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn log10(self) -> f16 {
        unsafe { intrinsics::log10f16(self) }
    }

    // /// Returns the cube root of a number.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = 8.0f16;
    // ///
    // /// // x^(1/3) - 2 == 0
    // /// let abs_difference = (x.cbrt() - 2.0).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn cbrt(self) -> f16 {
    //     unsafe { cmath::cbrtf(self) }
    // }

    // /// Compute the distance between the origin and a point (`x`, `y`) on the
    // /// Euclidean plane. Equivalently, compute the length of the hypotenuse of a
    // /// right-angle triangle with other sides having length `x.abs()` and
    // /// `y.abs()`.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = 2.0f16;
    // /// let y = 3.0f16;
    // ///
    // /// // sqrt(x^2 + y^2)
    // /// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn hypot(self, other: f16) -> f16 {
    //     unsafe { cmath::hypotf(self, other) }
    // }

    /// Computes the sine of a number (in radians).
    ///
    /// # Examples
    ///
    /// ```
    /// let x = std::f16::consts::FRAC_PI_2;
    ///
    /// let abs_difference = (x.sin() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn sin(self) -> f16 {
        unsafe { intrinsics::sinf16(self) }
    }

    /// Computes the cosine of a number (in radians).
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 2.0 * std::f16::consts::PI;
    ///
    /// let abs_difference = (x.cos() - 1.0).abs();
    ///
    /// assert!(abs_difference <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn cos(self) -> f16 {
        unsafe { intrinsics::cosf16(self) }
    }

    // /// Computes the tangent of a number (in radians).
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = std::f16::consts::FRAC_PI_4;
    // /// let abs_difference = (x.tan() - 1.0).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn tan(self) -> f16 {
    //     unsafe { cmath::tanf(self) }
    // }

    // /// Computes the arcsine of a number. Return value is in radians in
    // /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    // /// [-1, 1].
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let f = std::f16::consts::FRAC_PI_2;
    // ///
    // /// // asin(sin(pi/2))
    // /// let abs_difference = (f.sin().asin() - std::f16::consts::FRAC_PI_2).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn asin(self) -> f16 {
    //     unsafe { cmath::asinf(self) }
    // }

    // /// Computes the arccosine of a number. Return value is in radians in
    // /// the range [0, pi] or NaN if the number is outside the range
    // /// [-1, 1].
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let f = std::f16::consts::FRAC_PI_4;
    // ///
    // /// // acos(cos(pi/4))
    // /// let abs_difference = (f.cos().acos() - std::f16::consts::FRAC_PI_4).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn acos(self) -> f16 {
    //     unsafe { cmath::acosf(self) }
    // }

    // /// Computes the arctangent of a number. Return value is in radians in the
    // /// range [-pi/2, pi/2];
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let f = 1.0f16;
    // ///
    // /// // atan(tan(1))
    // /// let abs_difference = (f.tan().atan() - 1.0).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn atan(self) -> f16 {
    //     unsafe { cmath::atanf(self) }
    // }

    // /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    // ///
    // /// * `x = 0`, `y = 0`: `0`
    // /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    // /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    // /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// // Positive angles measured counter-clockwise
    // /// // from positive x axis
    // /// // -pi/4 radians (45 deg clockwise)
    // /// let x1 = 3.0f16;
    // /// let y1 = -3.0f16;
    // ///
    // /// // 3pi/4 radians (135 deg counter-clockwise)
    // /// let x2 = -3.0f16;
    // /// let y2 = 3.0f16;
    // ///
    // /// let abs_difference_1 = (y1.atan2(x1) - (-std::f16::consts::FRAC_PI_4)).abs();
    // /// let abs_difference_2 = (y2.atan2(x2) - (3.0 * std::f16::consts::FRAC_PI_4)).abs();
    // ///
    // /// assert!(abs_difference_1 <= f16::EPSILON);
    // /// assert!(abs_difference_2 <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn atan2(self, other: f16) -> f16 {
    //     unsafe { cmath::atan2f(self, other) }
    // }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = std::f16::consts::FRAC_PI_4;
    /// let f = x.sin_cos();
    ///
    /// let abs_difference_0 = (f.0 - x.sin()).abs();
    /// let abs_difference_1 = (f.1 - x.cos()).abs();
    ///
    /// assert!(abs_difference_0 <= f16::EPSILON);
    /// assert!(abs_difference_1 <= f16::EPSILON);
    /// ```
    #[inline]
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "f16_math", issue = "116909")]
    pub fn sin_cos(self) -> (f16, f16) {
        (self.sin(), self.cos())
    }

    // /// Returns `e^(self) - 1` in a way that is accurate even if the
    // /// number is close to zero.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = 1e-8_f16;
    // ///
    // /// // for very small x, e^x is approximately 1 + x + x^2 / 2
    // /// let approx = x + x * x / 2.0;
    // /// let abs_difference = (x.exp_m1() - approx).abs();
    // ///
    // /// assert!(abs_difference < 1e-10);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // pub fn exp_m1(self) -> f16 {
    //     unsafe { cmath::expm1f(self) }
    // }

    // /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    // /// the operations were performed separately.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = 1e-8_f16;
    // ///
    // /// // for very small x, ln(1 + x) is approximately x - x^2 / 2
    // /// let approx = x - x * x / 2.0;
    // /// let abs_difference = (x.ln_1p() - approx).abs();
    // ///
    // /// assert!(abs_difference < 1e-10);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // pub fn ln_1p(self) -> f16 {
    //     unsafe { cmath::log1pf(self) }
    // }

    // /// Hyperbolic sine function.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let e = std::f16::consts::E;
    // /// let x = 1.0f16;
    // ///
    // /// let f = x.sinh();
    // /// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
    // /// let g = ((e * e) - 1.0) / (2.0 * e);
    // /// let abs_difference = (f - g).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // pub fn sinh(self) -> f16 {
    //     unsafe { cmath::sinhf(self) }
    // }

    // /// Hyperbolic cosine function.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let e = std::f16::consts::E;
    // /// let x = 1.0f16;
    // /// let f = x.cosh();
    // /// // Solving cosh() at 1 gives this result
    // /// let g = ((e * e) + 1.0) / (2.0 * e);
    // /// let abs_difference = (f - g).abs();
    // ///
    // /// // Same result
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // pub fn cosh(self) -> f16 {
    //     unsafe { cmath::coshf(self) }
    // }

    // /// Hyperbolic tangent function.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let e = std::f16::consts::E;
    // /// let x = 1.0f16;
    // ///
    // /// let f = x.tanh();
    // /// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
    // /// let g = (1.0 - e.powi(-2)) / (1.0 + e.powi(-2));
    // /// let abs_difference = (f - g).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // pub fn tanh(self) -> f16 {
    //     unsafe { cmath::tanhf(self) }
    // }

    // /// Inverse hyperbolic sine function.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = 1.0f16;
    // /// let f = x.sinh().asinh();
    // ///
    // /// let abs_difference = (f - x).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn asinh(self) -> f16 {
    //     let ax = self.abs();
    //     let ix = 1.0 / ax;
    //     (ax + (ax / (Self::hypot(1.0, ix) + ix))).ln_1p().copysign(self)
    // }

    // /// Inverse hyperbolic cosine function.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let x = 1.0f16;
    // /// let f = x.cosh().acosh();
    // ///
    // /// let abs_difference = (f - x).abs();
    // ///
    // /// assert!(abs_difference <= f16::EPSILON);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn acosh(self) -> f16 {
    //     if self < 1.0 {
    //         Self::NAN
    //     } else {
    //         (self + ((self - 1.0).sqrt() * (self + 1.0).sqrt())).ln()
    //     }
    // }

    // /// Inverse hyperbolic tangent function.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// let e = std::f16::consts::E;
    // /// let f = e.tanh().atanh();
    // ///
    // /// let abs_difference = (f - e).abs();
    // ///
    // /// assert!(abs_difference <= 1e-5);
    // /// ```
    // #[inline]
    // #[rustc_allow_incoherent_impl]
    // #[unstable(feature = "f16_math", issue = "116909")]
    // #[must_use = "method returns a new number and does not mutate the original value"]
    // pub fn atanh(self) -> f16 {
    //     0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    // }
}

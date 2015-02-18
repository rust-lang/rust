// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Numeric traits and functions for generic mathematics
//!
//! These are implemented for the primitive numeric types in `std::{u8, u16,
//! u32, u64, uint, i8, i16, i32, i64, int, f32, f64}`.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

#[cfg(test)] use fmt::Debug;
use ops::{Add, Sub, Mul, Div, Rem, Neg};

use marker::Copy;
use clone::Clone;
use cmp::{PartialOrd, PartialEq};

pub use core::num::{Int, SignedInt, UnsignedInt};
pub use core::num::{cast, FromPrimitive, NumCast, ToPrimitive};
pub use core::num::{from_int, from_i8, from_i16, from_i32, from_i64};
pub use core::num::{from_uint, from_u8, from_u16, from_u32, from_u64};
pub use core::num::{from_f32, from_f64};
pub use core::num::{FromStrRadix, from_str_radix};
pub use core::num::{FpCategory, ParseIntError, ParseFloatError};

use option::Option;

#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub mod strconv;

/// Mathematical operations on primitive floating point numbers.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Float
    : Copy + Clone
    + NumCast
    + PartialOrd
    + PartialEq
    + Neg<Output=Self>
    + Add<Output=Self>
    + Sub<Output=Self>
    + Mul<Output=Self>
    + Div<Output=Self>
    + Rem<Output=Self>
{
    // inlined methods from `num::Float`
    /// Returns the `NaN` value.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let nan: f32 = Float::nan();
    ///
    /// assert!(nan.is_nan());
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn nan() -> Self;
    /// Returns the infinite value.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let infinity: f32 = Float::infinity();
    ///
    /// assert!(infinity.is_infinite());
    /// assert!(!infinity.is_finite());
    /// assert!(infinity > f32::MAX);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn infinity() -> Self;
    /// Returns the negative infinite value.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let neg_infinity: f32 = Float::neg_infinity();
    ///
    /// assert!(neg_infinity.is_infinite());
    /// assert!(!neg_infinity.is_finite());
    /// assert!(neg_infinity < f32::MIN);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn neg_infinity() -> Self;
    /// Returns `0.0`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let inf: f32 = Float::infinity();
    /// let zero: f32 = Float::zero();
    /// let neg_zero: f32 = Float::neg_zero();
    ///
    /// assert_eq!(zero, neg_zero);
    /// assert_eq!(7.0f32/inf, zero);
    /// assert_eq!(zero * 10.0, zero);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn zero() -> Self;
    /// Returns `-0.0`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let inf: f32 = Float::infinity();
    /// let zero: f32 = Float::zero();
    /// let neg_zero: f32 = Float::neg_zero();
    ///
    /// assert_eq!(zero, neg_zero);
    /// assert_eq!(7.0f32/inf, zero);
    /// assert_eq!(zero * 10.0, zero);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn neg_zero() -> Self;
    /// Returns `1.0`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let one: f32 = Float::one();
    ///
    /// assert_eq!(one, 1.0f32);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn one() -> Self;

    // FIXME (#5527): These should be associated constants

    /// Deprecated: use `std::f32::MANTISSA_DIGITS` or `std::f64::MANTISSA_DIGITS`
    /// instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MANTISSA_DIGITS` or \
                           `std::f64::MANTISSA_DIGITS` as appropriate")]
    fn mantissa_digits(unused_self: Option<Self>) -> uint;
    /// Deprecated: use `std::f32::DIGITS` or `std::f64::DIGITS` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::DIGITS` or `std::f64::DIGITS` as appropriate")]
    fn digits(unused_self: Option<Self>) -> uint;
    /// Deprecated: use `std::f32::EPSILON` or `std::f64::EPSILON` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::EPSILON` or `std::f64::EPSILON` as appropriate")]
    fn epsilon() -> Self;
    /// Deprecated: use `std::f32::MIN_EXP` or `std::f64::MIN_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MIN_EXP` or `std::f64::MIN_EXP` as appropriate")]
    fn min_exp(unused_self: Option<Self>) -> int;
    /// Deprecated: use `std::f32::MAX_EXP` or `std::f64::MAX_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MAX_EXP` or `std::f64::MAX_EXP` as appropriate")]
    fn max_exp(unused_self: Option<Self>) -> int;
    /// Deprecated: use `std::f32::MIN_10_EXP` or `std::f64::MIN_10_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MIN_10_EXP` or `std::f64::MIN_10_EXP` as appropriate")]
    fn min_10_exp(unused_self: Option<Self>) -> int;
    /// Deprecated: use `std::f32::MAX_10_EXP` or `std::f64::MAX_10_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MAX_10_EXP` or `std::f64::MAX_10_EXP` as appropriate")]
    fn max_10_exp(unused_self: Option<Self>) -> int;

    /// Returns the smallest finite value that this type can represent.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::min_value();
    ///
    /// assert_eq!(x, f64::MIN);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn min_value() -> Self;
    /// Returns the smallest normalized positive number that this type can represent.
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn min_pos_value(unused_self: Option<Self>) -> Self;
    /// Returns the largest finite value that this type can represent.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::max_value();
    /// assert_eq!(x, f64::MAX);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn max_value() -> Self;
    /// Returns `true` if this value is `NaN` and false otherwise.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let nan = f64::NAN;
    /// let f = 7.0;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    fn is_nan(self) -> bool;
    /// Returns `true` if this value is positive infinity or negative infinity and
    /// false otherwise.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf: f32 = Float::infinity();
    /// let neg_inf: f32 = Float::neg_infinity();
    /// let nan: f32 = f32::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    fn is_infinite(self) -> bool;
    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf: f32 = Float::infinity();
    /// let neg_inf: f32 = Float::neg_infinity();
    /// let nan: f32 = f32::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    fn is_finite(self) -> bool;
    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal][subnormal], or `NaN`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let min = f32::MIN_POSITIVE; // 1.17549435e-38f32
    /// let max = f32::MAX;
    /// let lower_than_min = 1.0e-40_f32;
    /// let zero = 0.0f32;
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!f32::NAN.is_normal());
    /// assert!(!f32::INFINITY.is_normal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    /// [subnormal]: http://en.wikipedia.org/wiki/Denormal_number
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    fn is_normal(self) -> bool;

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    /// use std::num::{Float, FpCategory};
    /// use std::f32;
    ///
    /// let num = 12.4f32;
    /// let inf = f32::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn classify(self) -> FpCategory;

    /// Returns the mantissa, base 2 exponent, and sign as integers, respectively.
    /// The original number can be recovered by `sign * mantissa * 2 ^ exponent`.
    /// The floating point encoding is documented in the [Reference][floating-point].
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let num = 2.0f32;
    ///
    /// // (8388608u64, -22i16, 1i8)
    /// let (mantissa, exponent, sign) = num.integer_decode();
    /// let sign_f = sign as f32;
    /// let mantissa_f = mantissa as f32;
    /// let exponent_f = num.powf(exponent as f32);
    ///
    /// // 1 * 8388608 * 2^(-22) == 2
    /// let abs_difference = (sign_f * mantissa_f * exponent_f - num).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    /// [floating-point]: ../../../../../reference.html#machine-types
    #[unstable(feature = "std_misc", reason = "signature is undecided")]
    fn integer_decode(self) -> (u64, i16, i8);

    /// Returns the largest integer less than or equal to a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.99;
    /// let g = 3.0;
    ///
    /// assert_eq!(f.floor(), 3.0);
    /// assert_eq!(g.floor(), 3.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn floor(self) -> Self;
    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.01;
    /// let g = 4.0;
    ///
    /// assert_eq!(f.ceil(), 4.0);
    /// assert_eq!(g.ceil(), 4.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ceil(self) -> Self;
    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.3;
    /// let g = -3.3;
    ///
    /// assert_eq!(f.round(), 3.0);
    /// assert_eq!(g.round(), -3.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn round(self) -> Self;
    /// Return the integer part of a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.3;
    /// let g = -3.7;
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), -3.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trunc(self) -> Self;
    /// Returns the fractional part of a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 3.5;
    /// let y = -3.5;
    /// let abs_difference_x = (x.fract() - 0.5).abs();
    /// let abs_difference_y = (y.fract() - (-0.5)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fract(self) -> Self;
    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = 3.5;
    /// let y = -3.5;
    ///
    /// let abs_difference_x = (x.abs() - x).abs();
    /// let abs_difference_y = (y.abs() - (-y)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    ///
    /// assert!(f64::NAN.abs().is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn abs(self) -> Self;
    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let f = 3.5;
    ///
    /// assert_eq!(f.signum(), 1.0);
    /// assert_eq!(f64::NEG_INFINITY.signum(), -1.0);
    ///
    /// assert!(f64::NAN.signum().is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn signum(self) -> Self;
    /// Returns `true` if `self` is positive, including `+0.0` and
    /// `Float::infinity()`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let nan: f64 = f64::NAN;
    ///
    /// let f = 7.0;
    /// let g = -7.0;
    ///
    /// assert!(f.is_positive());
    /// assert!(!g.is_positive());
    /// // Requires both tests to determine if is `NaN`
    /// assert!(!nan.is_positive() && !nan.is_negative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_positive(self) -> bool;
    /// Returns `true` if `self` is negative, including `-0.0` and
    /// `Float::neg_infinity()`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let nan = f64::NAN;
    ///
    /// let f = 7.0;
    /// let g = -7.0;
    ///
    /// assert!(!f.is_negative());
    /// assert!(g.is_negative());
    /// // Requires both tests to determine if is `NaN`.
    /// assert!(!nan.is_positive() && !nan.is_negative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_negative(self) -> bool;

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result with better performance than
    /// a separate multiplication operation followed by an add.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let m = 10.0;
    /// let x = 4.0;
    /// let b = 60.0;
    ///
    /// // 100.0
    /// let abs_difference = (m.mul_add(x, b) - (m*x + b)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Take the reciprocal (inverse) of a number, `1/x`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.recip() - (1.0/x)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn recip(self) -> Self;

    /// Raise a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.powi(2) - x*x).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn powi(self, n: i32) -> Self;
    /// Raise a number to a floating point power.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.powf(2.0) - x*x).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn powf(self, n: Self) -> Self;
    /// Take the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let positive = 4.0;
    /// let negative = -4.0;
    ///
    /// let abs_difference = (positive.sqrt() - 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// assert!(negative.sqrt().is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sqrt(self) -> Self;

    /// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 4.0;
    ///
    /// let abs_difference = (f.rsqrt() - 0.5).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn rsqrt(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let one = 1.0;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn exp(self) -> Self;
    /// Returns `2^(self)`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 2.0;
    ///
    /// // 2^2 - 4 == 0
    /// let abs_difference = (f.exp2() - 4.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn exp2(self) -> Self;
    /// Returns the natural logarithm of the number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let one = 1.0;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ln(self) -> Self;
    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let ten = 10.0;
    /// let two = 2.0;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference_10 = (ten.log(10.0) - 1.0).abs();
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference_2 = (two.log(2.0) - 1.0).abs();
    ///
    /// assert!(abs_difference_10 < 1e-10);
    /// assert!(abs_difference_2 < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn log(self, base: Self) -> Self;
    /// Returns the base 2 logarithm of the number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let two = 2.0;
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference = (two.log2() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn log2(self) -> Self;
    /// Returns the base 10 logarithm of the number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let ten = 10.0;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference = (ten.log10() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn log10(self) -> Self;

    /// Convert radians to degrees.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64::consts;
    ///
    /// let angle = consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "desirability is unclear")]
    fn to_degrees(self) -> Self;
    /// Convert degrees to radians.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64::consts;
    ///
    /// let angle = 180.0;
    ///
    /// let abs_difference = (angle.to_radians() - consts::PI).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "desirability is unclear")]
    fn to_radians(self) -> Self;
    /// Constructs a floating point number of `x*2^exp`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// // 3*2^2 - 12 == 0
    /// let abs_difference = (Float::ldexp(3.0, 2) - 12.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "pending integer conventions")]
    fn ldexp(x: Self, exp: int) -> Self;
    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    ///  * `self = x * 2^exp`
    ///  * `0.5 <= abs(x) < 1.0`
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 4.0;
    ///
    /// // (1/2)*2^3 -> 1 * 8/2 -> 4.0
    /// let f = x.frexp();
    /// let abs_difference_0 = (f.0 - 0.5).abs();
    /// let abs_difference_1 = (f.1 as f64 - 3.0).abs();
    ///
    /// assert!(abs_difference_0 < 1e-10);
    /// assert!(abs_difference_1 < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "pending integer conventions")]
    fn frexp(self) -> (Self, int);
    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0f32;
    ///
    /// let abs_diff = (x.next_after(2.0) - 1.00000011920928955078125_f32).abs();
    ///
    /// assert!(abs_diff < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn next_after(self, other: Self) -> Self;

    /// Returns the maximum of the two numbers.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let y = 2.0;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn max(self, other: Self) -> Self;
    /// Returns the minimum of the two numbers.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let y = 2.0;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn min(self, other: Self) -> Self;

    /// The positive difference of two numbers.
    ///
    /// * If `self <= other`: `0:0`
    /// * Else: `self - other`
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 3.0;
    /// let y = -3.0;
    ///
    /// let abs_difference_x = (x.abs_sub(1.0) - 2.0).abs();
    /// let abs_difference_y = (y.abs_sub(1.0) - 0.0).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    fn abs_sub(self, other: Self) -> Self;
    /// Take the cubic root of a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 8.0;
    ///
    /// // x^(1/3) - 2 == 0
    /// let abs_difference = (x.cbrt() - 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    fn cbrt(self) -> Self;
    /// Calculate the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let y = 3.0;
    ///
    /// // sqrt(x^2 + y^2)
    /// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    fn hypot(self, other: Self) -> Self;

    /// Computes the sine of a number (in radians).
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/2.0;
    ///
    /// let abs_difference = (x.sin() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sin(self) -> Self;
    /// Computes the cosine of a number (in radians).
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = 2.0*f64::consts::PI;
    ///
    /// let abs_difference = (x.cos() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn cos(self) -> Self;
    /// Computes the tangent of a number (in radians).
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/4.0;
    /// let abs_difference = (x.tan() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-14);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn tan(self) -> Self;
    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let f = f64::consts::PI / 2.0;
    ///
    /// // asin(sin(pi/2))
    /// let abs_difference = (f.sin().asin() - f64::consts::PI / 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn asin(self) -> Self;
    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let f = f64::consts::PI / 4.0;
    ///
    /// // acos(cos(pi/4))
    /// let abs_difference = (f.cos().acos() - f64::consts::PI / 4.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn acos(self) -> Self;
    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 1.0;
    ///
    /// // atan(tan(1))
    /// let abs_difference = (f.tan().atan() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn atan(self) -> Self;
    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`).
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let pi = f64::consts::PI;
    /// // All angles from horizontal right (+x)
    /// // 45 deg counter-clockwise
    /// let x1 = 3.0;
    /// let y1 = -3.0;
    ///
    /// // 135 deg clockwise
    /// let x2 = -3.0;
    /// let y2 = 3.0;
    ///
    /// let abs_difference_1 = (y1.atan2(x1) - (-pi/4.0)).abs();
    /// let abs_difference_2 = (y2.atan2(x2) - 3.0*pi/4.0).abs();
    ///
    /// assert!(abs_difference_1 < 1e-10);
    /// assert!(abs_difference_2 < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn atan2(self, other: Self) -> Self;
    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/4.0;
    /// let f = x.sin_cos();
    ///
    /// let abs_difference_0 = (f.0 - x.sin()).abs();
    /// let abs_difference_1 = (f.1 - x.cos()).abs();
    ///
    /// assert!(abs_difference_0 < 1e-10);
    /// assert!(abs_difference_0 < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sin_cos(self) -> (Self, Self);

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 7.0;
    ///
    /// // e^(ln(7)) - 1
    /// let abs_difference = (x.ln().exp_m1() - 6.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    fn exp_m1(self) -> Self;
    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::E - 1.0;
    ///
    /// // ln(1 + (e - 1)) == ln(e) == 1
    /// let abs_difference = (x.ln_1p() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    fn ln_1p(self) -> Self;

    /// Hyperbolic sine function.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    ///
    /// let f = x.sinh();
    /// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
    /// let g = (e*e - 1.0)/(2.0*e);
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sinh(self) -> Self;
    /// Hyperbolic cosine function.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    /// let f = x.cosh();
    /// // Solving cosh() at 1 gives this result
    /// let g = (e*e + 1.0)/(2.0*e);
    /// let abs_difference = (f - g).abs();
    ///
    /// // Same result
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn cosh(self) -> Self;
    /// Hyperbolic tangent function.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    ///
    /// let f = x.tanh();
    /// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
    /// let g = (1.0 - e.powi(-2))/(1.0 + e.powi(-2));
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn tanh(self) -> Self;
    /// Inverse hyperbolic sine function.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let f = x.sinh().asinh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine function.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let f = x.cosh().acosh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent function.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let f = e.tanh().atanh();
    ///
    /// let abs_difference = (f - e).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn atanh(self) -> Self;
}

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T>(ten: T, two: T) where
    T: PartialEq + NumCast
     + Add<Output=T> + Sub<Output=T>
     + Mul<Output=T> + Div<Output=T>
     + Rem<Output=T> + Debug
     + Copy
{
    assert_eq!(ten.add(two),  cast(12).unwrap());
    assert_eq!(ten.sub(two),  cast(8).unwrap());
    assert_eq!(ten.mul(two),  cast(20).unwrap());
    assert_eq!(ten.div(two),  cast(5).unwrap());
    assert_eq!(ten.rem(two),  cast(0).unwrap());

    assert_eq!(ten.add(two),  ten + two);
    assert_eq!(ten.sub(two),  ten - two);
    assert_eq!(ten.mul(two),  ten * two);
    assert_eq!(ten.div(two),  ten / two);
    assert_eq!(ten.rem(two),  ten % two);
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use super::*;
    use i8;
    use i16;
    use i32;
    use i64;
    use int;
    use u8;
    use u16;
    use u32;
    use u64;
    use uint;

    macro_rules! test_cast_20 {
        ($_20:expr) => ({
            let _20 = $_20;

            assert_eq!(20usize, _20.to_uint().unwrap());
            assert_eq!(20u8,    _20.to_u8().unwrap());
            assert_eq!(20u16,   _20.to_u16().unwrap());
            assert_eq!(20u32,   _20.to_u32().unwrap());
            assert_eq!(20u64,   _20.to_u64().unwrap());
            assert_eq!(20,      _20.to_int().unwrap());
            assert_eq!(20i8,    _20.to_i8().unwrap());
            assert_eq!(20i16,   _20.to_i16().unwrap());
            assert_eq!(20i32,   _20.to_i32().unwrap());
            assert_eq!(20i64,   _20.to_i64().unwrap());
            assert_eq!(20f32,   _20.to_f32().unwrap());
            assert_eq!(20f64,   _20.to_f64().unwrap());

            assert_eq!(_20, NumCast::from(20usize).unwrap());
            assert_eq!(_20, NumCast::from(20u8).unwrap());
            assert_eq!(_20, NumCast::from(20u16).unwrap());
            assert_eq!(_20, NumCast::from(20u32).unwrap());
            assert_eq!(_20, NumCast::from(20u64).unwrap());
            assert_eq!(_20, NumCast::from(20).unwrap());
            assert_eq!(_20, NumCast::from(20i8).unwrap());
            assert_eq!(_20, NumCast::from(20i16).unwrap());
            assert_eq!(_20, NumCast::from(20i32).unwrap());
            assert_eq!(_20, NumCast::from(20i64).unwrap());
            assert_eq!(_20, NumCast::from(20f32).unwrap());
            assert_eq!(_20, NumCast::from(20f64).unwrap());

            assert_eq!(_20, cast(20usize).unwrap());
            assert_eq!(_20, cast(20u8).unwrap());
            assert_eq!(_20, cast(20u16).unwrap());
            assert_eq!(_20, cast(20u32).unwrap());
            assert_eq!(_20, cast(20u64).unwrap());
            assert_eq!(_20, cast(20).unwrap());
            assert_eq!(_20, cast(20i8).unwrap());
            assert_eq!(_20, cast(20i16).unwrap());
            assert_eq!(_20, cast(20i32).unwrap());
            assert_eq!(_20, cast(20i64).unwrap());
            assert_eq!(_20, cast(20f32).unwrap());
            assert_eq!(_20, cast(20f64).unwrap());
        })
    }

    #[test] fn test_u8_cast()    { test_cast_20!(20u8)    }
    #[test] fn test_u16_cast()   { test_cast_20!(20u16)   }
    #[test] fn test_u32_cast()   { test_cast_20!(20u32)   }
    #[test] fn test_u64_cast()   { test_cast_20!(20u64)   }
    #[test] fn test_uint_cast()  { test_cast_20!(20usize) }
    #[test] fn test_i8_cast()    { test_cast_20!(20i8)    }
    #[test] fn test_i16_cast()   { test_cast_20!(20i16)   }
    #[test] fn test_i32_cast()   { test_cast_20!(20i32)   }
    #[test] fn test_i64_cast()   { test_cast_20!(20i64)   }
    #[test] fn test_int_cast()   { test_cast_20!(20)      }
    #[test] fn test_f32_cast()   { test_cast_20!(20f32)   }
    #[test] fn test_f64_cast()   { test_cast_20!(20f64)   }

    #[test]
    fn test_cast_range_int_min() {
        assert_eq!(int::MIN.to_int(),  Some(int::MIN as int));
        assert_eq!(int::MIN.to_i8(),   None);
        assert_eq!(int::MIN.to_i16(),  None);
        // int::MIN.to_i32() is word-size specific
        assert_eq!(int::MIN.to_i64(),  Some(int::MIN as i64));
        assert_eq!(int::MIN.to_uint(), None);
        assert_eq!(int::MIN.to_u8(),   None);
        assert_eq!(int::MIN.to_u16(),  None);
        assert_eq!(int::MIN.to_u32(),  None);
        assert_eq!(int::MIN.to_u64(),  None);

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(int::MIN.to_i32(), Some(int::MIN as i32));
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(int::MIN.to_i32(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_i8_min() {
        assert_eq!(i8::MIN.to_int(),  Some(i8::MIN as int));
        assert_eq!(i8::MIN.to_i8(),   Some(i8::MIN as i8));
        assert_eq!(i8::MIN.to_i16(),  Some(i8::MIN as i16));
        assert_eq!(i8::MIN.to_i32(),  Some(i8::MIN as i32));
        assert_eq!(i8::MIN.to_i64(),  Some(i8::MIN as i64));
        assert_eq!(i8::MIN.to_uint(), None);
        assert_eq!(i8::MIN.to_u8(),   None);
        assert_eq!(i8::MIN.to_u16(),  None);
        assert_eq!(i8::MIN.to_u32(),  None);
        assert_eq!(i8::MIN.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i16_min() {
        assert_eq!(i16::MIN.to_int(),  Some(i16::MIN as int));
        assert_eq!(i16::MIN.to_i8(),   None);
        assert_eq!(i16::MIN.to_i16(),  Some(i16::MIN as i16));
        assert_eq!(i16::MIN.to_i32(),  Some(i16::MIN as i32));
        assert_eq!(i16::MIN.to_i64(),  Some(i16::MIN as i64));
        assert_eq!(i16::MIN.to_uint(), None);
        assert_eq!(i16::MIN.to_u8(),   None);
        assert_eq!(i16::MIN.to_u16(),  None);
        assert_eq!(i16::MIN.to_u32(),  None);
        assert_eq!(i16::MIN.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i32_min() {
        assert_eq!(i32::MIN.to_int(),  Some(i32::MIN as int));
        assert_eq!(i32::MIN.to_i8(),   None);
        assert_eq!(i32::MIN.to_i16(),  None);
        assert_eq!(i32::MIN.to_i32(),  Some(i32::MIN as i32));
        assert_eq!(i32::MIN.to_i64(),  Some(i32::MIN as i64));
        assert_eq!(i32::MIN.to_uint(), None);
        assert_eq!(i32::MIN.to_u8(),   None);
        assert_eq!(i32::MIN.to_u16(),  None);
        assert_eq!(i32::MIN.to_u32(),  None);
        assert_eq!(i32::MIN.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i64_min() {
        // i64::MIN.to_int() is word-size specific
        assert_eq!(i64::MIN.to_i8(),   None);
        assert_eq!(i64::MIN.to_i16(),  None);
        assert_eq!(i64::MIN.to_i32(),  None);
        assert_eq!(i64::MIN.to_i64(),  Some(i64::MIN as i64));
        assert_eq!(i64::MIN.to_uint(), None);
        assert_eq!(i64::MIN.to_u8(),   None);
        assert_eq!(i64::MIN.to_u16(),  None);
        assert_eq!(i64::MIN.to_u32(),  None);
        assert_eq!(i64::MIN.to_u64(),  None);

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(i64::MIN.to_int(), None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(i64::MIN.to_int(), Some(i64::MIN as int));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_int_max() {
        assert_eq!(int::MAX.to_int(),  Some(int::MAX as int));
        assert_eq!(int::MAX.to_i8(),   None);
        assert_eq!(int::MAX.to_i16(),  None);
        // int::MAX.to_i32() is word-size specific
        assert_eq!(int::MAX.to_i64(),  Some(int::MAX as i64));
        assert_eq!(int::MAX.to_u8(),   None);
        assert_eq!(int::MAX.to_u16(),  None);
        // int::MAX.to_u32() is word-size specific
        assert_eq!(int::MAX.to_u64(),  Some(int::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(int::MAX.to_i32(), Some(int::MAX as i32));
            assert_eq!(int::MAX.to_u32(), Some(int::MAX as u32));
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(int::MAX.to_i32(), None);
            assert_eq!(int::MAX.to_u32(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_i8_max() {
        assert_eq!(i8::MAX.to_int(),  Some(i8::MAX as int));
        assert_eq!(i8::MAX.to_i8(),   Some(i8::MAX as i8));
        assert_eq!(i8::MAX.to_i16(),  Some(i8::MAX as i16));
        assert_eq!(i8::MAX.to_i32(),  Some(i8::MAX as i32));
        assert_eq!(i8::MAX.to_i64(),  Some(i8::MAX as i64));
        assert_eq!(i8::MAX.to_uint(), Some(i8::MAX as uint));
        assert_eq!(i8::MAX.to_u8(),   Some(i8::MAX as u8));
        assert_eq!(i8::MAX.to_u16(),  Some(i8::MAX as u16));
        assert_eq!(i8::MAX.to_u32(),  Some(i8::MAX as u32));
        assert_eq!(i8::MAX.to_u64(),  Some(i8::MAX as u64));
    }

    #[test]
    fn test_cast_range_i16_max() {
        assert_eq!(i16::MAX.to_int(),  Some(i16::MAX as int));
        assert_eq!(i16::MAX.to_i8(),   None);
        assert_eq!(i16::MAX.to_i16(),  Some(i16::MAX as i16));
        assert_eq!(i16::MAX.to_i32(),  Some(i16::MAX as i32));
        assert_eq!(i16::MAX.to_i64(),  Some(i16::MAX as i64));
        assert_eq!(i16::MAX.to_uint(), Some(i16::MAX as uint));
        assert_eq!(i16::MAX.to_u8(),   None);
        assert_eq!(i16::MAX.to_u16(),  Some(i16::MAX as u16));
        assert_eq!(i16::MAX.to_u32(),  Some(i16::MAX as u32));
        assert_eq!(i16::MAX.to_u64(),  Some(i16::MAX as u64));
    }

    #[test]
    fn test_cast_range_i32_max() {
        assert_eq!(i32::MAX.to_int(),  Some(i32::MAX as int));
        assert_eq!(i32::MAX.to_i8(),   None);
        assert_eq!(i32::MAX.to_i16(),  None);
        assert_eq!(i32::MAX.to_i32(),  Some(i32::MAX as i32));
        assert_eq!(i32::MAX.to_i64(),  Some(i32::MAX as i64));
        assert_eq!(i32::MAX.to_uint(), Some(i32::MAX as uint));
        assert_eq!(i32::MAX.to_u8(),   None);
        assert_eq!(i32::MAX.to_u16(),  None);
        assert_eq!(i32::MAX.to_u32(),  Some(i32::MAX as u32));
        assert_eq!(i32::MAX.to_u64(),  Some(i32::MAX as u64));
    }

    #[test]
    fn test_cast_range_i64_max() {
        // i64::MAX.to_int() is word-size specific
        assert_eq!(i64::MAX.to_i8(),   None);
        assert_eq!(i64::MAX.to_i16(),  None);
        assert_eq!(i64::MAX.to_i32(),  None);
        assert_eq!(i64::MAX.to_i64(),  Some(i64::MAX as i64));
        // i64::MAX.to_uint() is word-size specific
        assert_eq!(i64::MAX.to_u8(),   None);
        assert_eq!(i64::MAX.to_u16(),  None);
        assert_eq!(i64::MAX.to_u32(),  None);
        assert_eq!(i64::MAX.to_u64(),  Some(i64::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(i64::MAX.to_int(),  None);
            assert_eq!(i64::MAX.to_uint(), None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(i64::MAX.to_int(),  Some(i64::MAX as int));
            assert_eq!(i64::MAX.to_uint(), Some(i64::MAX as uint));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_uint_min() {
        assert_eq!(uint::MIN.to_int(),  Some(uint::MIN as int));
        assert_eq!(uint::MIN.to_i8(),   Some(uint::MIN as i8));
        assert_eq!(uint::MIN.to_i16(),  Some(uint::MIN as i16));
        assert_eq!(uint::MIN.to_i32(),  Some(uint::MIN as i32));
        assert_eq!(uint::MIN.to_i64(),  Some(uint::MIN as i64));
        assert_eq!(uint::MIN.to_uint(), Some(uint::MIN as uint));
        assert_eq!(uint::MIN.to_u8(),   Some(uint::MIN as u8));
        assert_eq!(uint::MIN.to_u16(),  Some(uint::MIN as u16));
        assert_eq!(uint::MIN.to_u32(),  Some(uint::MIN as u32));
        assert_eq!(uint::MIN.to_u64(),  Some(uint::MIN as u64));
    }

    #[test]
    fn test_cast_range_u8_min() {
        assert_eq!(u8::MIN.to_int(),  Some(u8::MIN as int));
        assert_eq!(u8::MIN.to_i8(),   Some(u8::MIN as i8));
        assert_eq!(u8::MIN.to_i16(),  Some(u8::MIN as i16));
        assert_eq!(u8::MIN.to_i32(),  Some(u8::MIN as i32));
        assert_eq!(u8::MIN.to_i64(),  Some(u8::MIN as i64));
        assert_eq!(u8::MIN.to_uint(), Some(u8::MIN as uint));
        assert_eq!(u8::MIN.to_u8(),   Some(u8::MIN as u8));
        assert_eq!(u8::MIN.to_u16(),  Some(u8::MIN as u16));
        assert_eq!(u8::MIN.to_u32(),  Some(u8::MIN as u32));
        assert_eq!(u8::MIN.to_u64(),  Some(u8::MIN as u64));
    }

    #[test]
    fn test_cast_range_u16_min() {
        assert_eq!(u16::MIN.to_int(),  Some(u16::MIN as int));
        assert_eq!(u16::MIN.to_i8(),   Some(u16::MIN as i8));
        assert_eq!(u16::MIN.to_i16(),  Some(u16::MIN as i16));
        assert_eq!(u16::MIN.to_i32(),  Some(u16::MIN as i32));
        assert_eq!(u16::MIN.to_i64(),  Some(u16::MIN as i64));
        assert_eq!(u16::MIN.to_uint(), Some(u16::MIN as uint));
        assert_eq!(u16::MIN.to_u8(),   Some(u16::MIN as u8));
        assert_eq!(u16::MIN.to_u16(),  Some(u16::MIN as u16));
        assert_eq!(u16::MIN.to_u32(),  Some(u16::MIN as u32));
        assert_eq!(u16::MIN.to_u64(),  Some(u16::MIN as u64));
    }

    #[test]
    fn test_cast_range_u32_min() {
        assert_eq!(u32::MIN.to_int(),  Some(u32::MIN as int));
        assert_eq!(u32::MIN.to_i8(),   Some(u32::MIN as i8));
        assert_eq!(u32::MIN.to_i16(),  Some(u32::MIN as i16));
        assert_eq!(u32::MIN.to_i32(),  Some(u32::MIN as i32));
        assert_eq!(u32::MIN.to_i64(),  Some(u32::MIN as i64));
        assert_eq!(u32::MIN.to_uint(), Some(u32::MIN as uint));
        assert_eq!(u32::MIN.to_u8(),   Some(u32::MIN as u8));
        assert_eq!(u32::MIN.to_u16(),  Some(u32::MIN as u16));
        assert_eq!(u32::MIN.to_u32(),  Some(u32::MIN as u32));
        assert_eq!(u32::MIN.to_u64(),  Some(u32::MIN as u64));
    }

    #[test]
    fn test_cast_range_u64_min() {
        assert_eq!(u64::MIN.to_int(),  Some(u64::MIN as int));
        assert_eq!(u64::MIN.to_i8(),   Some(u64::MIN as i8));
        assert_eq!(u64::MIN.to_i16(),  Some(u64::MIN as i16));
        assert_eq!(u64::MIN.to_i32(),  Some(u64::MIN as i32));
        assert_eq!(u64::MIN.to_i64(),  Some(u64::MIN as i64));
        assert_eq!(u64::MIN.to_uint(), Some(u64::MIN as uint));
        assert_eq!(u64::MIN.to_u8(),   Some(u64::MIN as u8));
        assert_eq!(u64::MIN.to_u16(),  Some(u64::MIN as u16));
        assert_eq!(u64::MIN.to_u32(),  Some(u64::MIN as u32));
        assert_eq!(u64::MIN.to_u64(),  Some(u64::MIN as u64));
    }

    #[test]
    fn test_cast_range_uint_max() {
        assert_eq!(uint::MAX.to_int(),  None);
        assert_eq!(uint::MAX.to_i8(),   None);
        assert_eq!(uint::MAX.to_i16(),  None);
        assert_eq!(uint::MAX.to_i32(),  None);
        // uint::MAX.to_i64() is word-size specific
        assert_eq!(uint::MAX.to_u8(),   None);
        assert_eq!(uint::MAX.to_u16(),  None);
        // uint::MAX.to_u32() is word-size specific
        assert_eq!(uint::MAX.to_u64(),  Some(uint::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(uint::MAX.to_u32(), Some(uint::MAX as u32));
            assert_eq!(uint::MAX.to_i64(), Some(uint::MAX as i64));
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(uint::MAX.to_u32(), None);
            assert_eq!(uint::MAX.to_i64(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_u8_max() {
        assert_eq!(u8::MAX.to_int(),  Some(u8::MAX as int));
        assert_eq!(u8::MAX.to_i8(),   None);
        assert_eq!(u8::MAX.to_i16(),  Some(u8::MAX as i16));
        assert_eq!(u8::MAX.to_i32(),  Some(u8::MAX as i32));
        assert_eq!(u8::MAX.to_i64(),  Some(u8::MAX as i64));
        assert_eq!(u8::MAX.to_uint(), Some(u8::MAX as uint));
        assert_eq!(u8::MAX.to_u8(),   Some(u8::MAX as u8));
        assert_eq!(u8::MAX.to_u16(),  Some(u8::MAX as u16));
        assert_eq!(u8::MAX.to_u32(),  Some(u8::MAX as u32));
        assert_eq!(u8::MAX.to_u64(),  Some(u8::MAX as u64));
    }

    #[test]
    fn test_cast_range_u16_max() {
        assert_eq!(u16::MAX.to_int(),  Some(u16::MAX as int));
        assert_eq!(u16::MAX.to_i8(),   None);
        assert_eq!(u16::MAX.to_i16(),  None);
        assert_eq!(u16::MAX.to_i32(),  Some(u16::MAX as i32));
        assert_eq!(u16::MAX.to_i64(),  Some(u16::MAX as i64));
        assert_eq!(u16::MAX.to_uint(), Some(u16::MAX as uint));
        assert_eq!(u16::MAX.to_u8(),   None);
        assert_eq!(u16::MAX.to_u16(),  Some(u16::MAX as u16));
        assert_eq!(u16::MAX.to_u32(),  Some(u16::MAX as u32));
        assert_eq!(u16::MAX.to_u64(),  Some(u16::MAX as u64));
    }

    #[test]
    fn test_cast_range_u32_max() {
        // u32::MAX.to_int() is word-size specific
        assert_eq!(u32::MAX.to_i8(),   None);
        assert_eq!(u32::MAX.to_i16(),  None);
        assert_eq!(u32::MAX.to_i32(),  None);
        assert_eq!(u32::MAX.to_i64(),  Some(u32::MAX as i64));
        assert_eq!(u32::MAX.to_uint(), Some(u32::MAX as uint));
        assert_eq!(u32::MAX.to_u8(),   None);
        assert_eq!(u32::MAX.to_u16(),  None);
        assert_eq!(u32::MAX.to_u32(),  Some(u32::MAX as u32));
        assert_eq!(u32::MAX.to_u64(),  Some(u32::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(u32::MAX.to_int(),  None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(u32::MAX.to_int(),  Some(u32::MAX as int));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_u64_max() {
        assert_eq!(u64::MAX.to_int(),  None);
        assert_eq!(u64::MAX.to_i8(),   None);
        assert_eq!(u64::MAX.to_i16(),  None);
        assert_eq!(u64::MAX.to_i32(),  None);
        assert_eq!(u64::MAX.to_i64(),  None);
        // u64::MAX.to_uint() is word-size specific
        assert_eq!(u64::MAX.to_u8(),   None);
        assert_eq!(u64::MAX.to_u16(),  None);
        assert_eq!(u64::MAX.to_u32(),  None);
        assert_eq!(u64::MAX.to_u64(),  Some(u64::MAX as u64));

        #[cfg(target_pointer_width = "32")]
        fn check_word_size() {
            assert_eq!(u64::MAX.to_uint(), None);
        }

        #[cfg(target_pointer_width = "64")]
        fn check_word_size() {
            assert_eq!(u64::MAX.to_uint(), Some(u64::MAX as uint));
        }

        check_word_size();
    }

    #[test]
    fn test_saturating_add_uint() {
        use uint::MAX;
        assert_eq!(3_usize.saturating_add(5_usize), 8_usize);
        assert_eq!(3_usize.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
    }

    #[test]
    fn test_saturating_sub_uint() {
        use uint::MAX;
        assert_eq!(5_usize.saturating_sub(3_usize), 2_usize);
        assert_eq!(3_usize.saturating_sub(5_usize), 0_usize);
        assert_eq!(0_usize.saturating_sub(1_usize), 0_usize);
        assert_eq!((MAX-1).saturating_sub(MAX), 0);
    }

    #[test]
    fn test_saturating_add_int() {
        use int::{MIN,MAX};
        assert_eq!(3.saturating_add(5), 8);
        assert_eq!(3.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
        assert_eq!(3.saturating_add(-5), -2);
        assert_eq!(MIN.saturating_add(-1), MIN);
        assert_eq!((-2).saturating_add(-MAX), MIN);
    }

    #[test]
    fn test_saturating_sub_int() {
        use int::{MIN,MAX};
        assert_eq!(3.saturating_sub(5), -2);
        assert_eq!(MIN.saturating_sub(1), MIN);
        assert_eq!((-2).saturating_sub(MAX), MIN);
        assert_eq!(3.saturating_sub(-5), 8);
        assert_eq!(3.saturating_sub(-(MAX-1)), MAX);
        assert_eq!(MAX.saturating_sub(-MAX), MAX);
        assert_eq!((MAX-2).saturating_sub(-1), MAX-1);
    }

    #[test]
    fn test_checked_add() {
        let five_less = uint::MAX - 5;
        assert_eq!(five_less.checked_add(0), Some(uint::MAX - 5));
        assert_eq!(five_less.checked_add(1), Some(uint::MAX - 4));
        assert_eq!(five_less.checked_add(2), Some(uint::MAX - 3));
        assert_eq!(five_less.checked_add(3), Some(uint::MAX - 2));
        assert_eq!(five_less.checked_add(4), Some(uint::MAX - 1));
        assert_eq!(five_less.checked_add(5), Some(uint::MAX));
        assert_eq!(five_less.checked_add(6), None);
        assert_eq!(five_less.checked_add(7), None);
    }

    #[test]
    fn test_checked_sub() {
        assert_eq!(5_usize.checked_sub(0), Some(5));
        assert_eq!(5_usize.checked_sub(1), Some(4));
        assert_eq!(5_usize.checked_sub(2), Some(3));
        assert_eq!(5_usize.checked_sub(3), Some(2));
        assert_eq!(5_usize.checked_sub(4), Some(1));
        assert_eq!(5_usize.checked_sub(5), Some(0));
        assert_eq!(5_usize.checked_sub(6), None);
        assert_eq!(5_usize.checked_sub(7), None);
    }

    #[test]
    fn test_checked_mul() {
        let third = uint::MAX / 3;
        assert_eq!(third.checked_mul(0), Some(0));
        assert_eq!(third.checked_mul(1), Some(third));
        assert_eq!(third.checked_mul(2), Some(third * 2));
        assert_eq!(third.checked_mul(3), Some(third * 3));
        assert_eq!(third.checked_mul(4), None);
    }

    macro_rules! test_is_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).is_power_of_two(), false);
                assert_eq!((1 as $T).is_power_of_two(), true);
                assert_eq!((2 as $T).is_power_of_two(), true);
                assert_eq!((3 as $T).is_power_of_two(), false);
                assert_eq!((4 as $T).is_power_of_two(), true);
                assert_eq!((5 as $T).is_power_of_two(), false);
                assert!(($T::MAX / 2 + 1).is_power_of_two(), true);
            }
        )
    }

    test_is_power_of_two!{ test_is_power_of_two_u8, u8 }
    test_is_power_of_two!{ test_is_power_of_two_u16, u16 }
    test_is_power_of_two!{ test_is_power_of_two_u32, u32 }
    test_is_power_of_two!{ test_is_power_of_two_u64, u64 }
    test_is_power_of_two!{ test_is_power_of_two_uint, uint }

    macro_rules! test_next_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).next_power_of_two(), 1);
                let mut next_power = 1;
                for i in range::<$T>(1, 40) {
                     assert_eq!(i.next_power_of_two(), next_power);
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    }

    test_next_power_of_two! { test_next_power_of_two_u8, u8 }
    test_next_power_of_two! { test_next_power_of_two_u16, u16 }
    test_next_power_of_two! { test_next_power_of_two_u32, u32 }
    test_next_power_of_two! { test_next_power_of_two_u64, u64 }
    test_next_power_of_two! { test_next_power_of_two_uint, uint }

    macro_rules! test_checked_next_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).checked_next_power_of_two(), Some(1));
                assert!(($T::MAX / 2).checked_next_power_of_two().is_some());
                assert_eq!(($T::MAX - 1).checked_next_power_of_two(), None);
                assert_eq!($T::MAX.checked_next_power_of_two(), None);
                let mut next_power = 1;
                for i in range::<$T>(1, 40) {
                     assert_eq!(i.checked_next_power_of_two(), Some(next_power));
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    }

    test_checked_next_power_of_two! { test_checked_next_power_of_two_u8, u8 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u16, u16 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u32, u32 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u64, u64 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_uint, uint }

    #[derive(PartialEq, Debug)]
    struct Value { x: int }

    impl ToPrimitive for Value {
        fn to_i64(&self) -> Option<i64> { self.x.to_i64() }
        fn to_u64(&self) -> Option<u64> { self.x.to_u64() }
    }

    impl FromPrimitive for Value {
        fn from_i64(n: i64) -> Option<Value> { Some(Value { x: n as int }) }
        fn from_u64(n: u64) -> Option<Value> { Some(Value { x: n as int }) }
    }

    #[test]
    fn test_to_primitive() {
        let value = Value { x: 5 };
        assert_eq!(value.to_int(),  Some(5));
        assert_eq!(value.to_i8(),   Some(5));
        assert_eq!(value.to_i16(),  Some(5));
        assert_eq!(value.to_i32(),  Some(5));
        assert_eq!(value.to_i64(),  Some(5));
        assert_eq!(value.to_uint(), Some(5));
        assert_eq!(value.to_u8(),   Some(5));
        assert_eq!(value.to_u16(),  Some(5));
        assert_eq!(value.to_u32(),  Some(5));
        assert_eq!(value.to_u64(),  Some(5));
        assert_eq!(value.to_f32(),  Some(5f32));
        assert_eq!(value.to_f64(),  Some(5f64));
    }

    #[test]
    fn test_from_primitive() {
        assert_eq!(from_int(5),    Some(Value { x: 5 }));
        assert_eq!(from_i8(5),     Some(Value { x: 5 }));
        assert_eq!(from_i16(5),    Some(Value { x: 5 }));
        assert_eq!(from_i32(5),    Some(Value { x: 5 }));
        assert_eq!(from_i64(5),    Some(Value { x: 5 }));
        assert_eq!(from_uint(5),   Some(Value { x: 5 }));
        assert_eq!(from_u8(5),     Some(Value { x: 5 }));
        assert_eq!(from_u16(5),    Some(Value { x: 5 }));
        assert_eq!(from_u32(5),    Some(Value { x: 5 }));
        assert_eq!(from_u64(5),    Some(Value { x: 5 }));
        assert_eq!(from_f32(5f32), Some(Value { x: 5 }));
        assert_eq!(from_f64(5f64), Some(Value { x: 5 }));
    }

    #[test]
    fn test_pow() {
        fn naive_pow<T: Int>(base: T, exp: uint) -> T {
            let one: T = Int::one();
            (0..exp).fold(one, |acc, _| acc * base)
        }
        macro_rules! assert_pow {
            (($num:expr, $exp:expr) => $expected:expr) => {{
                let result = $num.pow($exp);
                assert_eq!(result, $expected);
                assert_eq!(result, naive_pow($num, $exp));
            }}
        }
        assert_pow!((3,     0 ) => 1);
        assert_pow!((5,     1 ) => 5);
        assert_pow!((-4,    2 ) => 16);
        assert_pow!((8,     3 ) => 512);
        assert_pow!((2u64,   50) => 1125899906842624);
    }
}


#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use num::Int;
    use prelude::v1::*;

    #[bench]
    fn bench_pow_function(b: &mut Bencher) {
        let v = (0..1024).collect::<Vec<_>>();
        b.iter(|| {v.iter().fold(0, |old, new| old.pow(*new));});
    }
}

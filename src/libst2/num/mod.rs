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

#![stable]
#![allow(missing_docs)]

#[cfg(test)] use cmp::PartialEq;
#[cfg(test)] use fmt::Show;
#[cfg(test)] use ops::{Add, Sub, Mul, Div, Rem};

pub use core::num::{Num, div_rem, Zero, zero, One, one};
pub use core::num::{Unsigned, pow, Bounded};
pub use core::num::{Primitive, Int, SignedInt, UnsignedInt};
pub use core::num::{cast, FromPrimitive, NumCast, ToPrimitive};
pub use core::num::{next_power_of_two, is_power_of_two};
pub use core::num::{checked_next_power_of_two};
pub use core::num::{from_int, from_i8, from_i16, from_i32, from_i64};
pub use core::num::{from_uint, from_u8, from_u16, from_u32, from_u64};
pub use core::num::{from_f32, from_f64};
pub use core::num::{FromStrRadix, from_str_radix};
pub use core::num::{FPCategory, FPNaN, FPInfinite, FPZero, FPSubnormal};
pub use core::num::{FPNormal, Float};

#[experimental = "may be removed or relocated"]
pub mod strconv;

/// Mathematical operations on primitive floating point numbers.
#[unstable = "may be altered to inline the Float trait"]
pub trait FloatMath: Float {
    /// Constructs a floating point number created by multiplying `x` by 2
    /// raised to the power of `exp`.
    fn ldexp(x: Self, exp: int) -> Self;
    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    ///  * `self = x * pow(2, exp)`
    ///
    ///  * `0.5 <= abs(x) < 1.0`
    fn frexp(self) -> (Self, int);

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    fn next_after(self, other: Self) -> Self;

    /// Returns the maximum of the two numbers.
    fn max(self, other: Self) -> Self;
    /// Returns the minimum of the two numbers.
    fn min(self, other: Self) -> Self;

    /// The positive difference of two numbers. Returns `0.0` if the number is
    /// less than or equal to `other`, otherwise the difference between`self`
    /// and `other` is returned.
    fn abs_sub(self, other: Self) -> Self;

    /// Take the cubic root of a number.
    fn cbrt(self) -> Self;
    /// Calculate the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    fn hypot(self, other: Self) -> Self;

    /// Computes the sine of a number (in radians).
    fn sin(self) -> Self;
    /// Computes the cosine of a number (in radians).
    fn cos(self) -> Self;
    /// Computes the tangent of a number (in radians).
    fn tan(self) -> Self;

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    fn asin(self) -> Self;
    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    fn acos(self) -> Self;
    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    fn atan(self) -> Self;
    /// Computes the four quadrant arctangent of a number, `y`, and another
    /// number `x`. Return value is in radians in the range [-pi, pi].
    fn atan2(self, other: Self) -> Self;
    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    fn sin_cos(self) -> (Self, Self);

    /// Returns the exponential of the number, minus 1, in a way that is
    /// accurate even if the number is close to zero.
    fn exp_m1(self) -> Self;
    /// Returns the natural logarithm of the number plus 1 (`ln(1+n)`) more
    /// accurately than if the operations were performed separately.
    fn ln_1p(self) -> Self;

    /// Hyperbolic sine function.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine function.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent function.
    fn tanh(self) -> Self;
    /// Inverse hyperbolic sine function.
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine function.
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent function.
    fn atanh(self) -> Self;
}

// DEPRECATED

#[deprecated = "Use `FloatMath::abs_sub`"]
pub fn abs_sub<T: FloatMath>(x: T, y: T) -> T { unimplemented!() }

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T>(ten: T, two: T) where
    T: PartialEq + NumCast
     + Add<T, T> + Sub<T, T>
     + Mul<T, T> + Div<T, T>
     + Rem<T, T> + Show
{
    assert_eq!(ten.add(&two),  cast(12i).unwrap());
    assert_eq!(ten.sub(&two),  cast(8i).unwrap());
    assert_eq!(ten.mul(&two),  cast(20i).unwrap());
    assert_eq!(ten.div(&two),  cast(5i).unwrap());
    assert_eq!(ten.rem(&two),  cast(0i).unwrap());

    assert_eq!(ten.add(&two),  ten + two);
    assert_eq!(ten.sub(&two),  ten - two);
    assert_eq!(ten.mul(&two),  ten * two);
    assert_eq!(ten.div(&two),  ten / two);
    assert_eq!(ten.rem(&two),  ten % two);
}

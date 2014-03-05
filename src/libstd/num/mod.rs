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
//! u32, u64, uint, i8, i16, i32, i64, int, f32, f64, float}`.

#[allow(missing_doc)];

use clone::{Clone, DeepClone};
use cmp::{Eq, Ord};
use kinds::Pod;
use mem::size_of;
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use ops::{Not, BitAnd, BitOr, BitXor, Shl, Shr};
use option::{Option, Some, None};
use fmt::{Show, Binary, Octal, LowerHex, UpperHex};

pub mod strconv;

/// The base trait for numeric types
pub trait Num: Eq + Zero + One
             + Neg<Self>
             + Add<Self,Self>
             + Sub<Self,Self>
             + Mul<Self,Self>
             + Div<Self,Self>
             + Rem<Self,Self> {}

/// Simultaneous division and remainder
#[inline]
pub fn div_rem<T: Div<T, T> + Rem<T, T>>(x: T, y: T) -> (T, T) {
    (x / y, x % y)
}

/// Defines an additive identity element for `Self`.
///
/// # Deriving
///
/// This trait can be automatically be derived using `#[deriving(Zero)]`
/// attribute. If you choose to use this, make sure that the laws outlined in
/// the documentation for `Zero::zero` still hold.
pub trait Zero: Add<Self, Self> {
    /// Returns the additive identity element of `Self`, `0`.
    ///
    /// # Laws
    ///
    /// ~~~notrust
    /// a + 0 = a       ∀ a ∈ Self
    /// 0 + a = a       ∀ a ∈ Self
    /// ~~~
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // FIXME (#5527): This should be an associated constant
    fn zero() -> Self;

    /// Returns `true` if `self` is equal to the additive identity.
    fn is_zero(&self) -> bool;
}

/// Returns the additive identity, `0`.
#[inline(always)] pub fn zero<T: Zero>() -> T { Zero::zero() }

/// Defines a multiplicative identity element for `Self`.
pub trait One: Mul<Self, Self> {
    /// Returns the multiplicative identity element of `Self`, `1`.
    ///
    /// # Laws
    ///
    /// ~~~notrust
    /// a * 1 = a       ∀ a ∈ Self
    /// 1 * a = a       ∀ a ∈ Self
    /// ~~~
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // FIXME (#5527): This should be an associated constant
    fn one() -> Self;
}

/// Returns the multiplicative identity, `1`.
#[inline(always)] pub fn one<T: One>() -> T { One::one() }

pub trait Signed: Num
                + Neg<Self> {
    fn abs(&self) -> Self;
    fn abs_sub(&self, other: &Self) -> Self;
    fn signum(&self) -> Self;

    fn is_positive(&self) -> bool;
    fn is_negative(&self) -> bool;
}

/// Computes the absolute value.
///
/// For float, f32, and f64, `NaN` will be returned if the number is `NaN`
#[inline(always)] pub fn abs<T: Signed>(value: T) -> T { value.abs() }
/// The positive difference of two numbers.
///
/// Returns `zero` if the number is less than or equal to `other`,
/// otherwise the difference between `self` and `other` is returned.
#[inline(always)] pub fn abs_sub<T: Signed>(x: T, y: T) -> T { x.abs_sub(&y) }
/// Returns the sign of the number.
///
/// For float, f32, f64:
/// - `1.0` if the number is positive, `+0.0` or `INFINITY`
/// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
/// - `NAN` if the number is `NAN`
///
/// For int:
/// - `0` if the number is zero
/// - `1` if the number is positive
/// - `-1` if the number is negative
#[inline(always)] pub fn signum<T: Signed>(value: T) -> T { value.signum() }

pub trait Unsigned: Num {}

/// A collection of rounding operations.
pub trait Round {
    /// Return the largest integer less than or equal to a number.
    fn floor(&self) -> Self;

    /// Return the smallest integer greater than or equal to a number.
    fn ceil(&self) -> Self;

    /// Return the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    fn round(&self) -> Self;

    /// Return the integer part of a number.
    fn trunc(&self) -> Self;

    /// Return the fractional part of a number.
    fn fract(&self) -> Self;
}

/// Raises a value to the power of exp, using exponentiation by squaring.
///
/// # Example
///
/// ```rust
/// use std::num;
///
/// assert_eq!(num::pow(2, 4), 16);
/// ```
#[inline]
pub fn pow<T: One + Mul<T, T>>(mut base: T, mut exp: uint) -> T {
    if exp == 1 { base }
    else {
        let mut acc = one::<T>();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = base * base;
            exp = exp >> 1;
        }
        acc
    }
}

pub trait Bounded {
    // FIXME (#5527): These should be associated constants
    fn min_value() -> Self;
    fn max_value() -> Self;
}

/// Numbers with a fixed binary representation.
pub trait Bitwise: Bounded
                 + Not<Self>
                 + BitAnd<Self,Self>
                 + BitOr<Self,Self>
                 + BitXor<Self,Self>
                 + Shl<Self,Self>
                 + Shr<Self,Self> {
    /// Returns the number of ones in the binary representation of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b01001100u8;
    /// assert_eq!(n.count_ones(), 3);
    /// ```
    fn count_ones(&self) -> Self;

    /// Returns the number of zeros in the binary representation of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b01001100u8;
    /// assert_eq!(n.count_zeros(), 5);
    /// ```
    #[inline]
    fn count_zeros(&self) -> Self {
        (!*self).count_ones()
    }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b0101000u16;
    /// assert_eq!(n.leading_zeros(), 10);
    /// ```
    fn leading_zeros(&self) -> Self;

    /// Returns the number of trailing zeros in the in the binary representation
    /// of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b0101000u16;
    /// assert_eq!(n.trailing_zeros(), 3);
    /// ```
    fn trailing_zeros(&self) -> Self;
}

/// Specifies the available operations common to all of Rust's core numeric primitives.
/// These may not always make sense from a purely mathematical point of view, but
/// may be useful for systems programming.
pub trait Primitive: Pod
                   + Clone
                   + DeepClone
                   + Num
                   + NumCast
                   + Ord
                   + Bounded {}

/// A collection of traits relevant to primitive signed and unsigned integers
pub trait Int: Primitive
             + Bitwise
             + CheckedAdd
             + CheckedSub
             + CheckedMul
             + CheckedDiv
             + Show
             + Binary
             + Octal
             + LowerHex
             + UpperHex {}

/// Returns the smallest power of 2 greater than or equal to `n`.
#[inline]
pub fn next_power_of_two<T: Unsigned + Int>(n: T) -> T {
    let halfbits: T = cast(size_of::<T>() * 4).unwrap();
    let mut tmp: T = n - one();
    let mut shift: T = one();
    while shift <= halfbits {
        tmp = tmp | (tmp >> shift);
        shift = shift << one();
    }
    tmp + one()
}

/// Returns the smallest power of 2 greater than or equal to `n`. If the next
/// power of two is greater than the type's maximum value, `None` is returned,
/// otherwise the power of 2 is wrapped in `Some`.
#[inline]
pub fn checked_next_power_of_two<T: Unsigned + Int>(n: T) -> Option<T> {
    let halfbits: T = cast(size_of::<T>() * 4).unwrap();
    let mut tmp: T = n - one();
    let mut shift: T = one();
    while shift <= halfbits {
        tmp = tmp | (tmp >> shift);
        shift = shift << one();
    }
    tmp.checked_add(&one())
}

/// Used for representing the classification of floating point numbers
#[deriving(Eq, Show)]
pub enum FPCategory {
    /// "Not a Number", often obtained by dividing by zero
    FPNaN,
    /// Positive or negative infinity
    FPInfinite ,
    /// Positive or negative zero
    FPZero,
    /// De-normalized floating point representation (less precise than `FPNormal`)
    FPSubnormal,
    /// A regular floating point number
    FPNormal,
}

/// Primitive floating point numbers
pub trait Float: Signed
               + Round
               + Primitive {
    // FIXME (#5527): These should be associated constants
    fn nan() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn neg_zero() -> Self;

    fn is_nan(&self) -> bool;
    fn is_infinite(&self) -> bool;
    fn is_finite(&self) -> bool;
    fn is_normal(&self) -> bool;
    fn classify(&self) -> FPCategory;

    // FIXME (#8888): Removing `unused_self` requires #8888 to be fixed.
    fn mantissa_digits(unused_self: Option<Self>) -> uint;
    fn digits(unused_self: Option<Self>) -> uint;
    fn epsilon() -> Self;
    fn min_exp(unused_self: Option<Self>) -> int;
    fn max_exp(unused_self: Option<Self>) -> int;
    fn min_10_exp(unused_self: Option<Self>) -> int;
    fn max_10_exp(unused_self: Option<Self>) -> int;

    fn ldexp(x: Self, exp: int) -> Self;
    fn frexp(&self) -> (Self, int);

    fn exp_m1(&self) -> Self;
    fn ln_1p(&self) -> Self;
    fn mul_add(&self, a: Self, b: Self) -> Self;
    fn next_after(&self, other: Self) -> Self;

    fn integer_decode(&self) -> (u64, i16, i8);

    // Common Mathematical Constants
    // FIXME (#5527): These should be associated constants
    fn pi() -> Self;
    fn two_pi() -> Self;
    fn frac_pi_2() -> Self;
    fn frac_pi_3() -> Self;
    fn frac_pi_4() -> Self;
    fn frac_pi_6() -> Self;
    fn frac_pi_8() -> Self;
    fn frac_1_pi() -> Self;
    fn frac_2_pi() -> Self;
    fn frac_2_sqrtpi() -> Self;
    fn sqrt2() -> Self;
    fn frac_1_sqrt2() -> Self;
    fn e() -> Self;
    fn log2_e() -> Self;
    fn log10_e() -> Self;
    fn ln_2() -> Self;
    fn ln_10() -> Self;

    // Fractional functions

    /// Take the reciprocal (inverse) of a number, `1/x`.
    fn recip(&self) -> Self;

    // Algebraic functions
    /// Raise a number to a power.
    fn powf(&self, n: &Self) -> Self;

    /// Take the square root of a number.
    fn sqrt(&self) -> Self;
    /// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
    fn rsqrt(&self) -> Self;
    /// Take the cubic root of a number.
    fn cbrt(&self) -> Self;
    /// Calculate the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    fn hypot(&self, other: &Self) -> Self;

    // Trigonometric functions

    /// Computes the sine of a number (in radians).
    fn sin(&self) -> Self;
    /// Computes the cosine of a number (in radians).
    fn cos(&self) -> Self;
    /// Computes the tangent of a number (in radians).
    fn tan(&self) -> Self;

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    fn asin(&self) -> Self;
    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    fn acos(&self) -> Self;
    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    fn atan(&self) -> Self;
    /// Computes the four quadrant arctangent of a number, `y`, and another
    /// number `x`. Return value is in radians in the range [-pi, pi].
    fn atan2(&self, other: &Self) -> Self;
    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    fn sin_cos(&self) -> (Self, Self);

    // Exponential functions

    /// Returns `e^(self)`, (the exponential function).
    fn exp(&self) -> Self;
    /// Returns 2 raised to the power of the number, `2^(self)`.
    fn exp2(&self) -> Self;
    /// Returns the natural logarithm of the number.
    fn ln(&self) -> Self;
    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(&self, base: &Self) -> Self;
    /// Returns the base 2 logarithm of the number.
    fn log2(&self) -> Self;
    /// Returns the base 10 logarithm of the number.
    fn log10(&self) -> Self;

    // Hyperbolic functions

    /// Hyperbolic sine function.
    fn sinh(&self) -> Self;
    /// Hyperbolic cosine function.
    fn cosh(&self) -> Self;
    /// Hyperbolic tangent function.
    fn tanh(&self) -> Self;
    /// Inverse hyperbolic sine function.
    fn asinh(&self) -> Self;
    /// Inverse hyperbolic cosine function.
    fn acosh(&self) -> Self;
    /// Inverse hyperbolic tangent function.
    fn atanh(&self) -> Self;

    // Angular conversions

    /// Convert radians to degrees.
    fn to_degrees(&self) -> Self;
    /// Convert degrees to radians.
    fn to_radians(&self) -> Self;
}

/// Returns the exponential of the number, minus `1`, `exp(n) - 1`, in a way
/// that is accurate even if the number is close to zero.
#[inline(always)] pub fn exp_m1<T: Float>(value: T) -> T { value.exp_m1() }
/// Returns the natural logarithm of the number plus `1`, `ln(n + 1)`, more
/// accurately than if the operations were performed separately.
#[inline(always)] pub fn ln_1p<T: Float>(value: T) -> T { value.ln_1p() }
/// Fused multiply-add. Computes `(a * b) + c` with only one rounding error.
///
/// This produces a more accurate result with better performance (on some
/// architectures) than a separate multiplication operation followed by an add.
#[inline(always)] pub fn mul_add<T: Float>(a: T, b: T, c: T) -> T { a.mul_add(b, c) }

/// Raise a number to a power.
///
/// # Example
///
/// ```rust
/// use std::num;
///
/// let sixteen: f64 = num::powf(2.0, 4.0);
/// assert_eq!(sixteen, 16.0);
/// ```
#[inline(always)] pub fn powf<T: Float>(value: T, n: T) -> T { value.powf(&n) }
/// Take the square root of a number.
#[inline(always)] pub fn sqrt<T: Float>(value: T) -> T { value.sqrt() }
/// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
#[inline(always)] pub fn rsqrt<T: Float>(value: T) -> T { value.rsqrt() }
/// Take the cubic root of a number.
#[inline(always)] pub fn cbrt<T: Float>(value: T) -> T { value.cbrt() }
/// Calculate the length of the hypotenuse of a right-angle triangle given legs
/// of length `x` and `y`.
#[inline(always)] pub fn hypot<T: Float>(x: T, y: T) -> T { x.hypot(&y) }
/// Sine function.
#[inline(always)] pub fn sin<T: Float>(value: T) -> T { value.sin() }
/// Cosine function.
#[inline(always)] pub fn cos<T: Float>(value: T) -> T { value.cos() }
/// Tangent function.
#[inline(always)] pub fn tan<T: Float>(value: T) -> T { value.tan() }
/// Compute the arcsine of the number.
#[inline(always)] pub fn asin<T: Float>(value: T) -> T { value.asin() }
/// Compute the arccosine of the number.
#[inline(always)] pub fn acos<T: Float>(value: T) -> T { value.acos() }
/// Compute the arctangent of the number.
#[inline(always)] pub fn atan<T: Float>(value: T) -> T { value.atan() }
/// Compute the arctangent with 2 arguments.
#[inline(always)] pub fn atan2<T: Float>(x: T, y: T) -> T { x.atan2(&y) }
/// Simultaneously computes the sine and cosine of the number.
#[inline(always)] pub fn sin_cos<T: Float>(value: T) -> (T, T) { value.sin_cos() }
/// Returns `e^(value)`, (the exponential function).
#[inline(always)] pub fn exp<T: Float>(value: T) -> T { value.exp() }
/// Returns 2 raised to the power of the number, `2^(value)`.
#[inline(always)] pub fn exp2<T: Float>(value: T) -> T { value.exp2() }
/// Returns the natural logarithm of the number.
#[inline(always)] pub fn ln<T: Float>(value: T) -> T { value.ln() }
/// Returns the logarithm of the number with respect to an arbitrary base.
#[inline(always)] pub fn log<T: Float>(value: T, base: T) -> T { value.log(&base) }
/// Returns the base 2 logarithm of the number.
#[inline(always)] pub fn log2<T: Float>(value: T) -> T { value.log2() }
/// Returns the base 10 logarithm of the number.
#[inline(always)] pub fn log10<T: Float>(value: T) -> T { value.log10() }
/// Hyperbolic sine function.
#[inline(always)] pub fn sinh<T: Float>(value: T) -> T { value.sinh() }
/// Hyperbolic cosine function.
#[inline(always)] pub fn cosh<T: Float>(value: T) -> T { value.cosh() }
/// Hyperbolic tangent function.
#[inline(always)] pub fn tanh<T: Float>(value: T) -> T { value.tanh() }
/// Inverse hyperbolic sine function.
#[inline(always)] pub fn asinh<T: Float>(value: T) -> T { value.asinh() }
/// Inverse hyperbolic cosine function.
#[inline(always)] pub fn acosh<T: Float>(value: T) -> T { value.acosh() }
/// Inverse hyperbolic tangent function.
#[inline(always)] pub fn atanh<T: Float>(value: T) -> T { value.atanh() }

/// A generic trait for converting a value to a number.
pub trait ToPrimitive {
    /// Converts the value of `self` to an `int`.
    #[inline]
    fn to_int(&self) -> Option<int> {
        self.to_i64().and_then(|x| x.to_int())
    }

    /// Converts the value of `self` to an `i8`.
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.to_i64().and_then(|x| x.to_i8())
    }

    /// Converts the value of `self` to an `i16`.
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.to_i64().and_then(|x| x.to_i16())
    }

    /// Converts the value of `self` to an `i32`.
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.to_i64().and_then(|x| x.to_i32())
    }

    /// Converts the value of `self` to an `i64`.
    fn to_i64(&self) -> Option<i64>;

    /// Converts the value of `self` to an `uint`.
    #[inline]
    fn to_uint(&self) -> Option<uint> {
        self.to_u64().and_then(|x| x.to_uint())
    }

    /// Converts the value of `self` to an `u8`.
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.to_u64().and_then(|x| x.to_u8())
    }

    /// Converts the value of `self` to an `u16`.
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.to_u64().and_then(|x| x.to_u16())
    }

    /// Converts the value of `self` to an `u32`.
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.to_u64().and_then(|x| x.to_u32())
    }

    /// Converts the value of `self` to an `u64`.
    #[inline]
    fn to_u64(&self) -> Option<u64>;

    /// Converts the value of `self` to an `f32`.
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.to_f64().and_then(|x| x.to_f32())
    }

    /// Converts the value of `self` to an `f64`.
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.to_i64().and_then(|x| x.to_f64())
    }
}

macro_rules! impl_to_primitive_int_to_int(
    ($SrcT:ty, $DstT:ty) => (
        {
            if size_of::<$SrcT>() <= size_of::<$DstT>() {
                Some(*self as $DstT)
            } else {
                let n = *self as i64;
                let min_value: $DstT = Bounded::min_value();
                let max_value: $DstT = Bounded::max_value();
                if min_value as i64 <= n && n <= max_value as i64 {
                    Some(*self as $DstT)
                } else {
                    None
                }
            }
        }
    )
)

macro_rules! impl_to_primitive_int_to_uint(
    ($SrcT:ty, $DstT:ty) => (
        {
            let zero: $SrcT = Zero::zero();
            let max_value: $DstT = Bounded::max_value();
            if zero <= *self && *self as u64 <= max_value as u64 {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_int(
    ($T:ty) => (
        impl ToPrimitive for $T {
            #[inline]
            fn to_int(&self) -> Option<int> { impl_to_primitive_int_to_int!($T, int) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { impl_to_primitive_int_to_int!($T, i8) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { impl_to_primitive_int_to_int!($T, i16) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { impl_to_primitive_int_to_int!($T, i32) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { impl_to_primitive_int_to_int!($T, i64) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { impl_to_primitive_int_to_uint!($T, uint) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { impl_to_primitive_int_to_uint!($T, u8) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { impl_to_primitive_int_to_uint!($T, u16) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { impl_to_primitive_int_to_uint!($T, u32) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { impl_to_primitive_int_to_uint!($T, u64) }

            #[inline]
            fn to_f32(&self) -> Option<f32> { Some(*self as f32) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { Some(*self as f64) }
        }
    )
)

impl_to_primitive_int!(int)
impl_to_primitive_int!(i8)
impl_to_primitive_int!(i16)
impl_to_primitive_int!(i32)
impl_to_primitive_int!(i64)

macro_rules! impl_to_primitive_uint_to_int(
    ($DstT:ty) => (
        {
            let max_value: $DstT = Bounded::max_value();
            if *self as u64 <= max_value as u64 {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_uint_to_uint(
    ($SrcT:ty, $DstT:ty) => (
        {
            if size_of::<$SrcT>() <= size_of::<$DstT>() {
                Some(*self as $DstT)
            } else {
                let zero: $SrcT = Zero::zero();
                let max_value: $DstT = Bounded::max_value();
                if zero <= *self && *self as u64 <= max_value as u64 {
                    Some(*self as $DstT)
                } else {
                    None
                }
            }
        }
    )
)

macro_rules! impl_to_primitive_uint(
    ($T:ty) => (
        impl ToPrimitive for $T {
            #[inline]
            fn to_int(&self) -> Option<int> { impl_to_primitive_uint_to_int!(int) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { impl_to_primitive_uint_to_int!(i8) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { impl_to_primitive_uint_to_int!(i16) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { impl_to_primitive_uint_to_int!(i32) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { impl_to_primitive_uint_to_int!(i64) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { impl_to_primitive_uint_to_uint!($T, uint) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { impl_to_primitive_uint_to_uint!($T, u8) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { impl_to_primitive_uint_to_uint!($T, u16) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { impl_to_primitive_uint_to_uint!($T, u32) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { impl_to_primitive_uint_to_uint!($T, u64) }

            #[inline]
            fn to_f32(&self) -> Option<f32> { Some(*self as f32) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { Some(*self as f64) }
        }
    )
)

impl_to_primitive_uint!(uint)
impl_to_primitive_uint!(u8)
impl_to_primitive_uint!(u16)
impl_to_primitive_uint!(u32)
impl_to_primitive_uint!(u64)

macro_rules! impl_to_primitive_float_to_float(
    ($SrcT:ty, $DstT:ty) => (
        if size_of::<$SrcT>() <= size_of::<$DstT>() {
            Some(*self as $DstT)
        } else {
            let n = *self as f64;
            let max_value: $SrcT = Bounded::max_value();
            if -max_value as f64 <= n && n <= max_value as f64 {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_float(
    ($T:ty) => (
        impl ToPrimitive for $T {
            #[inline]
            fn to_int(&self) -> Option<int> { Some(*self as int) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { Some(*self as i8) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { Some(*self as i16) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { Some(*self as i32) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { Some(*self as i64) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { Some(*self as uint) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { Some(*self as u8) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { Some(*self as u16) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { Some(*self as u32) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { Some(*self as u64) }

            #[inline]
            fn to_f32(&self) -> Option<f32> { impl_to_primitive_float_to_float!($T, f32) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { impl_to_primitive_float_to_float!($T, f64) }
        }
    )
)

impl_to_primitive_float!(f32)
impl_to_primitive_float!(f64)

/// A generic trait for converting a number to a value.
pub trait FromPrimitive {
    /// Convert an `int` to return an optional value of this type. If the
    /// value cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_int(n: int) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i8` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_i8(n: i8) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i16` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_i16(n: i16) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i32` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_i32(n: i32) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i64` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    fn from_i64(n: i64) -> Option<Self>;

    /// Convert an `uint` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_uint(n: uint) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u8` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u16` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_u16(n: u16) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u32` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_u32(n: u32) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u64` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    fn from_u64(n: u64) -> Option<Self>;

    /// Convert a `f32` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        FromPrimitive::from_f64(n as f64)
    }

    /// Convert a `f64` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }
}

/// A utility function that just calls `FromPrimitive::from_int`.
pub fn from_int<A: FromPrimitive>(n: int) -> Option<A> {
    FromPrimitive::from_int(n)
}

/// A utility function that just calls `FromPrimitive::from_i8`.
pub fn from_i8<A: FromPrimitive>(n: i8) -> Option<A> {
    FromPrimitive::from_i8(n)
}

/// A utility function that just calls `FromPrimitive::from_i16`.
pub fn from_i16<A: FromPrimitive>(n: i16) -> Option<A> {
    FromPrimitive::from_i16(n)
}

/// A utility function that just calls `FromPrimitive::from_i32`.
pub fn from_i32<A: FromPrimitive>(n: i32) -> Option<A> {
    FromPrimitive::from_i32(n)
}

/// A utility function that just calls `FromPrimitive::from_i64`.
pub fn from_i64<A: FromPrimitive>(n: i64) -> Option<A> {
    FromPrimitive::from_i64(n)
}

/// A utility function that just calls `FromPrimitive::from_uint`.
pub fn from_uint<A: FromPrimitive>(n: uint) -> Option<A> {
    FromPrimitive::from_uint(n)
}

/// A utility function that just calls `FromPrimitive::from_u8`.
pub fn from_u8<A: FromPrimitive>(n: u8) -> Option<A> {
    FromPrimitive::from_u8(n)
}

/// A utility function that just calls `FromPrimitive::from_u16`.
pub fn from_u16<A: FromPrimitive>(n: u16) -> Option<A> {
    FromPrimitive::from_u16(n)
}

/// A utility function that just calls `FromPrimitive::from_u32`.
pub fn from_u32<A: FromPrimitive>(n: u32) -> Option<A> {
    FromPrimitive::from_u32(n)
}

/// A utility function that just calls `FromPrimitive::from_u64`.
pub fn from_u64<A: FromPrimitive>(n: u64) -> Option<A> {
    FromPrimitive::from_u64(n)
}

/// A utility function that just calls `FromPrimitive::from_f32`.
pub fn from_f32<A: FromPrimitive>(n: f32) -> Option<A> {
    FromPrimitive::from_f32(n)
}

/// A utility function that just calls `FromPrimitive::from_f64`.
pub fn from_f64<A: FromPrimitive>(n: f64) -> Option<A> {
    FromPrimitive::from_f64(n)
}

macro_rules! impl_from_primitive(
    ($T:ty, $to_ty:expr) => (
        impl FromPrimitive for $T {
            #[inline] fn from_int(n: int) -> Option<$T> { $to_ty }
            #[inline] fn from_i8(n: i8) -> Option<$T> { $to_ty }
            #[inline] fn from_i16(n: i16) -> Option<$T> { $to_ty }
            #[inline] fn from_i32(n: i32) -> Option<$T> { $to_ty }
            #[inline] fn from_i64(n: i64) -> Option<$T> { $to_ty }

            #[inline] fn from_uint(n: uint) -> Option<$T> { $to_ty }
            #[inline] fn from_u8(n: u8) -> Option<$T> { $to_ty }
            #[inline] fn from_u16(n: u16) -> Option<$T> { $to_ty }
            #[inline] fn from_u32(n: u32) -> Option<$T> { $to_ty }
            #[inline] fn from_u64(n: u64) -> Option<$T> { $to_ty }

            #[inline] fn from_f32(n: f32) -> Option<$T> { $to_ty }
            #[inline] fn from_f64(n: f64) -> Option<$T> { $to_ty }
        }
    )
)

impl_from_primitive!(int, n.to_int())
impl_from_primitive!(i8, n.to_i8())
impl_from_primitive!(i16, n.to_i16())
impl_from_primitive!(i32, n.to_i32())
impl_from_primitive!(i64, n.to_i64())
impl_from_primitive!(uint, n.to_uint())
impl_from_primitive!(u8, n.to_u8())
impl_from_primitive!(u16, n.to_u16())
impl_from_primitive!(u32, n.to_u32())
impl_from_primitive!(u64, n.to_u64())
impl_from_primitive!(f32, n.to_f32())
impl_from_primitive!(f64, n.to_f64())

/// Cast from one machine scalar to another.
///
/// # Example
///
/// ```
/// use std::num;
///
/// let twenty: f32 = num::cast(0x14).unwrap();
/// assert_eq!(twenty, 20f32);
/// ```
///
#[inline]
pub fn cast<T: NumCast,U: NumCast>(n: T) -> Option<U> {
    NumCast::from(n)
}

/// An interface for casting between machine scalars
pub trait NumCast: ToPrimitive {
    fn from<T: ToPrimitive>(n: T) -> Option<Self>;
}

macro_rules! impl_num_cast(
    ($T:ty, $conv:ident) => (
        impl NumCast for $T {
            #[inline]
            fn from<N: ToPrimitive>(n: N) -> Option<$T> {
                // `$conv` could be generated using `concat_idents!`, but that
                // macro seems to be broken at the moment
                n.$conv()
            }
        }
    )
)

impl_num_cast!(u8,    to_u8)
impl_num_cast!(u16,   to_u16)
impl_num_cast!(u32,   to_u32)
impl_num_cast!(u64,   to_u64)
impl_num_cast!(uint,  to_uint)
impl_num_cast!(i8,    to_i8)
impl_num_cast!(i16,   to_i16)
impl_num_cast!(i32,   to_i32)
impl_num_cast!(i64,   to_i64)
impl_num_cast!(int,   to_int)
impl_num_cast!(f32,   to_f32)
impl_num_cast!(f64,   to_f64)

pub trait ToStrRadix {
    fn to_str_radix(&self, radix: uint) -> ~str;
}

pub trait FromStrRadix {
    fn from_str_radix(str: &str, radix: uint) -> Option<Self>;
}

/// A utility function that just calls FromStrRadix::from_str_radix.
pub fn from_str_radix<T: FromStrRadix>(str: &str, radix: uint) -> Option<T> {
    FromStrRadix::from_str_radix(str, radix)
}

/// Saturating math operations
pub trait Saturating {
    /// Saturating addition operator.
    /// Returns a+b, saturating at the numeric bounds instead of overflowing.
    fn saturating_add(self, v: Self) -> Self;

    /// Saturating subtraction operator.
    /// Returns a-b, saturating at the numeric bounds instead of overflowing.
    fn saturating_sub(self, v: Self) -> Self;
}

impl<T: CheckedAdd + CheckedSub + Zero + Ord + Bounded> Saturating for T {
    #[inline]
    fn saturating_add(self, v: T) -> T {
        match self.checked_add(&v) {
            Some(x) => x,
            None => if v >= Zero::zero() {
                Bounded::max_value()
            } else {
                Bounded::min_value()
            }
        }
    }

    #[inline]
    fn saturating_sub(self, v: T) -> T {
        match self.checked_sub(&v) {
            Some(x) => x,
            None => if v >= Zero::zero() {
                Bounded::min_value()
            } else {
                Bounded::max_value()
            }
        }
    }
}

pub trait CheckedAdd: Add<Self, Self> {
    fn checked_add(&self, v: &Self) -> Option<Self>;
}

pub trait CheckedSub: Sub<Self, Self> {
    fn checked_sub(&self, v: &Self) -> Option<Self>;
}

pub trait CheckedMul: Mul<Self, Self> {
    fn checked_mul(&self, v: &Self) -> Option<Self>;
}

pub trait CheckedDiv: Div<Self, Self> {
    fn checked_div(&self, v: &Self) -> Option<Self>;
}

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T:Num + NumCast + Show>(ten: T, two: T) {
    assert_eq!(ten.add(&two),  cast(12).unwrap());
    assert_eq!(ten.sub(&two),  cast(8).unwrap());
    assert_eq!(ten.mul(&two),  cast(20).unwrap());
    assert_eq!(ten.div(&two),  cast(5).unwrap());
    assert_eq!(ten.rem(&two),  cast(0).unwrap());

    assert_eq!(ten.add(&two),  ten + two);
    assert_eq!(ten.sub(&two),  ten - two);
    assert_eq!(ten.mul(&two),  ten * two);
    assert_eq!(ten.div(&two),  ten / two);
    assert_eq!(ten.rem(&two),  ten % two);
}

#[cfg(test)]
mod tests {
    use prelude::*;
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

    macro_rules! test_cast_20(
        ($_20:expr) => ({
            let _20 = $_20;

            assert_eq!(20u,   _20.to_uint().unwrap());
            assert_eq!(20u8,  _20.to_u8().unwrap());
            assert_eq!(20u16, _20.to_u16().unwrap());
            assert_eq!(20u32, _20.to_u32().unwrap());
            assert_eq!(20u64, _20.to_u64().unwrap());
            assert_eq!(20i,   _20.to_int().unwrap());
            assert_eq!(20i8,  _20.to_i8().unwrap());
            assert_eq!(20i16, _20.to_i16().unwrap());
            assert_eq!(20i32, _20.to_i32().unwrap());
            assert_eq!(20i64, _20.to_i64().unwrap());
            assert_eq!(20f32, _20.to_f32().unwrap());
            assert_eq!(20f64, _20.to_f64().unwrap());

            assert_eq!(_20, NumCast::from(20u).unwrap());
            assert_eq!(_20, NumCast::from(20u8).unwrap());
            assert_eq!(_20, NumCast::from(20u16).unwrap());
            assert_eq!(_20, NumCast::from(20u32).unwrap());
            assert_eq!(_20, NumCast::from(20u64).unwrap());
            assert_eq!(_20, NumCast::from(20i).unwrap());
            assert_eq!(_20, NumCast::from(20i8).unwrap());
            assert_eq!(_20, NumCast::from(20i16).unwrap());
            assert_eq!(_20, NumCast::from(20i32).unwrap());
            assert_eq!(_20, NumCast::from(20i64).unwrap());
            assert_eq!(_20, NumCast::from(20f32).unwrap());
            assert_eq!(_20, NumCast::from(20f64).unwrap());

            assert_eq!(_20, cast(20u).unwrap());
            assert_eq!(_20, cast(20u8).unwrap());
            assert_eq!(_20, cast(20u16).unwrap());
            assert_eq!(_20, cast(20u32).unwrap());
            assert_eq!(_20, cast(20u64).unwrap());
            assert_eq!(_20, cast(20i).unwrap());
            assert_eq!(_20, cast(20i8).unwrap());
            assert_eq!(_20, cast(20i16).unwrap());
            assert_eq!(_20, cast(20i32).unwrap());
            assert_eq!(_20, cast(20i64).unwrap());
            assert_eq!(_20, cast(20f32).unwrap());
            assert_eq!(_20, cast(20f64).unwrap());
        })
    )

    #[test] fn test_u8_cast()    { test_cast_20!(20u8)  }
    #[test] fn test_u16_cast()   { test_cast_20!(20u16) }
    #[test] fn test_u32_cast()   { test_cast_20!(20u32) }
    #[test] fn test_u64_cast()   { test_cast_20!(20u64) }
    #[test] fn test_uint_cast()  { test_cast_20!(20u)   }
    #[test] fn test_i8_cast()    { test_cast_20!(20i8)  }
    #[test] fn test_i16_cast()   { test_cast_20!(20i16) }
    #[test] fn test_i32_cast()   { test_cast_20!(20i32) }
    #[test] fn test_i64_cast()   { test_cast_20!(20i64) }
    #[test] fn test_int_cast()   { test_cast_20!(20i)   }
    #[test] fn test_f32_cast()   { test_cast_20!(20f32) }
    #[test] fn test_f64_cast()   { test_cast_20!(20f64) }

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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(int::MIN.to_i32(), Some(int::MIN as i32));
        }

        #[cfg(target_word_size = "64")]
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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(i64::MIN.to_int(), None);
        }

        #[cfg(target_word_size = "64")]
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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(int::MAX.to_i32(), Some(int::MAX as i32));
            assert_eq!(int::MAX.to_u32(), Some(int::MAX as u32));
        }

        #[cfg(target_word_size = "64")]
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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(i64::MAX.to_int(),  None);
            assert_eq!(i64::MAX.to_uint(), None);
        }

        #[cfg(target_word_size = "64")]
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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(uint::MAX.to_u32(), Some(uint::MAX as u32));
            assert_eq!(uint::MAX.to_i64(), Some(uint::MAX as i64));
        }

        #[cfg(target_word_size = "64")]
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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(u32::MAX.to_int(),  None);
        }

        #[cfg(target_word_size = "64")]
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

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(u64::MAX.to_uint(), None);
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(u64::MAX.to_uint(), Some(u64::MAX as uint));
        }

        check_word_size();
    }

    #[test]
    fn test_saturating_add_uint() {
        use uint::MAX;
        assert_eq!(3u.saturating_add(5u), 8u);
        assert_eq!(3u.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
    }

    #[test]
    fn test_saturating_sub_uint() {
        use uint::MAX;
        assert_eq!(5u.saturating_sub(3u), 2u);
        assert_eq!(3u.saturating_sub(5u), 0u);
        assert_eq!(0u.saturating_sub(1u), 0u);
        assert_eq!((MAX-1).saturating_sub(MAX), 0);
    }

    #[test]
    fn test_saturating_add_int() {
        use int::{MIN,MAX};
        assert_eq!(3i.saturating_add(5i), 8i);
        assert_eq!(3i.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
        assert_eq!(3i.saturating_add(-5i), -2i);
        assert_eq!(MIN.saturating_add(-1i), MIN);
        assert_eq!((-2i).saturating_add(-MAX), MIN);
    }

    #[test]
    fn test_saturating_sub_int() {
        use int::{MIN,MAX};
        assert_eq!(3i.saturating_sub(5i), -2i);
        assert_eq!(MIN.saturating_sub(1i), MIN);
        assert_eq!((-2i).saturating_sub(MAX), MIN);
        assert_eq!(3i.saturating_sub(-5i), 8i);
        assert_eq!(3i.saturating_sub(-(MAX-1)), MAX);
        assert_eq!(MAX.saturating_sub(-MAX), MAX);
        assert_eq!((MAX-2).saturating_sub(-1), MAX-1);
    }

    #[test]
    fn test_checked_add() {
        let five_less = uint::MAX - 5;
        assert_eq!(five_less.checked_add(&0), Some(uint::MAX - 5));
        assert_eq!(five_less.checked_add(&1), Some(uint::MAX - 4));
        assert_eq!(five_less.checked_add(&2), Some(uint::MAX - 3));
        assert_eq!(five_less.checked_add(&3), Some(uint::MAX - 2));
        assert_eq!(five_less.checked_add(&4), Some(uint::MAX - 1));
        assert_eq!(five_less.checked_add(&5), Some(uint::MAX));
        assert_eq!(five_less.checked_add(&6), None);
        assert_eq!(five_less.checked_add(&7), None);
    }

    #[test]
    fn test_checked_sub() {
        assert_eq!(5u.checked_sub(&0), Some(5));
        assert_eq!(5u.checked_sub(&1), Some(4));
        assert_eq!(5u.checked_sub(&2), Some(3));
        assert_eq!(5u.checked_sub(&3), Some(2));
        assert_eq!(5u.checked_sub(&4), Some(1));
        assert_eq!(5u.checked_sub(&5), Some(0));
        assert_eq!(5u.checked_sub(&6), None);
        assert_eq!(5u.checked_sub(&7), None);
    }

    #[test]
    fn test_checked_mul() {
        let third = uint::MAX / 3;
        assert_eq!(third.checked_mul(&0), Some(0));
        assert_eq!(third.checked_mul(&1), Some(third));
        assert_eq!(third.checked_mul(&2), Some(third * 2));
        assert_eq!(third.checked_mul(&3), Some(third * 3));
        assert_eq!(third.checked_mul(&4), None);
    }

    macro_rules! test_next_power_of_two(
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #[test];
                assert_eq!(next_power_of_two::<$T>(0), 0);
                let mut next_power = 1;
                for i in range::<$T>(1, 40) {
                     assert_eq!(next_power_of_two(i), next_power);
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    )

    test_next_power_of_two!(test_next_power_of_two_u8, u8)
    test_next_power_of_two!(test_next_power_of_two_u16, u16)
    test_next_power_of_two!(test_next_power_of_two_u32, u32)
    test_next_power_of_two!(test_next_power_of_two_u64, u64)
    test_next_power_of_two!(test_next_power_of_two_uint, uint)

    macro_rules! test_checked_next_power_of_two(
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #[test];
                assert_eq!(checked_next_power_of_two::<$T>(0), None);
                let mut next_power = 1;
                for i in range::<$T>(1, 40) {
                     assert_eq!(checked_next_power_of_two(i), Some(next_power));
                     if i == next_power { next_power *= 2 }
                }
                assert!(checked_next_power_of_two::<$T>($T::MAX / 2).is_some());
                assert_eq!(checked_next_power_of_two::<$T>($T::MAX - 1), None);
                assert_eq!(checked_next_power_of_two::<$T>($T::MAX), None);
            }
        )
    )

    test_checked_next_power_of_two!(test_checked_next_power_of_two_u8, u8)
    test_checked_next_power_of_two!(test_checked_next_power_of_two_u16, u16)
    test_checked_next_power_of_two!(test_checked_next_power_of_two_u32, u32)
    test_checked_next_power_of_two!(test_checked_next_power_of_two_u64, u64)
    test_checked_next_power_of_two!(test_checked_next_power_of_two_uint, uint)

    #[deriving(Eq, Show)]
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
        fn naive_pow<T: One + Mul<T, T>>(base: T, exp: uint) -> T {
            range(0, exp).fold(one::<T>(), |acc, _| acc * base)
        }
        macro_rules! assert_pow(
            (($num:expr, $exp:expr) => $expected:expr) => {{
                let result = pow($num, $exp);
                assert_eq!(result, $expected);
                assert_eq!(result, naive_pow($num, $exp));
            }}
        )
        assert_pow!((3,    0 ) => 1);
        assert_pow!((5,    1 ) => 5);
        assert_pow!((-4,   2 ) => 16);
        assert_pow!((0.5,  5 ) => 0.03125);
        assert_pow!((8,    3 ) => 512);
        assert_pow!((8.0,  5 ) => 32768.0);
        assert_pow!((8.5,  5 ) => 44370.53125);
        assert_pow!((2u64, 50) => 1125899906842624);
    }
}


#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::BenchHarness;
    use num;
    use vec;
    use prelude::*;

    #[bench]
    fn bench_pow_function(b: &mut BenchHarness) {
        let v = vec::from_fn(1024, |n| n);
        b.iter(|| {v.iter().fold(0, |old, new| num::pow(old, *new));});
    }
}

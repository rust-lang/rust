// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Numeric traits and functions for generic mathematics.
//!
//! These are implemented for the primitive numeric types in `std::{u8, u16,
//! u32, u64, uint, i8, i16, i32, i64, int, f32, f64, float}`.

#[allow(missing_doc)];

use clone::{Clone, DeepClone};
use cmp::{Eq, ApproxEq, Ord};
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use ops::{Not, BitAnd, BitOr, BitXor, Shl, Shr};
use option::{Option, Some, None};

pub mod strconv;

/// The base trait for numeric types
pub trait Num: Eq + Zero + One
             + Neg<Self>
             + Add<Self,Self>
             + Sub<Self,Self>
             + Mul<Self,Self>
             + Div<Self,Self>
             + Rem<Self,Self> {}

pub trait Orderable: Ord {
    // These should be methods on `Ord`, with overridable default implementations. We don't want
    // to encumber all implementors of Ord by requiring them to implement these functions, but at
    // the same time we want to be able to take advantage of the speed of the specific numeric
    // functions (like the `fmin` and `fmax` intrinsics).
    fn min(&self, other: &Self) -> Self;
    fn max(&self, other: &Self) -> Self;
    fn clamp(&self, mn: &Self, mx: &Self) -> Self;
}

/// Return the smaller number.
#[inline(always)] pub fn min<T: Orderable>(x: T, y: T) -> T { x.min(&y) }
/// Return the larger number.
#[inline(always)] pub fn max<T: Orderable>(x: T, y: T) -> T { x.max(&y) }
/// Returns the number constrained within the range `mn <= self <= mx`.
#[inline(always)] pub fn clamp<T: Orderable>(value: T, mn: T, mx: T) -> T { value.clamp(&mn, &mx) }

pub trait Zero {
    fn zero() -> Self;      // FIXME (#5527): This should be an associated constant
    fn is_zero(&self) -> bool;
}

/// Returns `0` of appropriate type.
#[inline(always)] pub fn zero<T: Zero>() -> T { Zero::zero() }

pub trait One {
    fn one() -> Self;       // FIXME (#5527): This should be an associated constant
}

/// Returns `1` of appropriate type.
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
/// - `1.0` if the number is positive, `+0.0` or `infinity`
/// - `-1.0` if the number is negative, `-0.0` or `neg_infinity`
/// - `NaN` if the number is `NaN`
///
/// For int:
/// - `0` if the number is zero
/// - `1` if the number is positive
/// - `-1` if the number is negative
#[inline(always)] pub fn signum<T: Signed>(value: T) -> T { value.signum() }

pub trait Unsigned: Num {}

/// Times trait
///
/// ```rust
/// use num::Times;
/// let ten = 10 as uint;
/// let mut accum = 0;
/// do ten.times { accum += 1; }
/// ```
///
pub trait Times {
    fn times(&self, it: &fn());
}

pub trait Integer: Num
                 + Orderable
                 + Div<Self,Self>
                 + Rem<Self,Self> {
    fn div_rem(&self, other: &Self) -> (Self,Self);

    fn div_floor(&self, other: &Self) -> Self;
    fn mod_floor(&self, other: &Self) -> Self;
    fn div_mod_floor(&self, other: &Self) -> (Self,Self);

    fn gcd(&self, other: &Self) -> Self;
    fn lcm(&self, other: &Self) -> Self;

    fn is_multiple_of(&self, other: &Self) -> bool;
    fn is_even(&self) -> bool;
    fn is_odd(&self) -> bool;
}

/// Calculates the Greatest Common Divisor (GCD) of the number and `other`.
///
/// The result is always positive.
#[inline(always)] pub fn gcd<T: Integer>(x: T, y: T) -> T { x.gcd(&y) }
/// Calculates the Lowest Common Multiple (LCM) of the number and `other`.
#[inline(always)] pub fn lcm<T: Integer>(x: T, y: T) -> T { x.lcm(&y) }

pub trait Round {
    fn floor(&self) -> Self;
    fn ceil(&self) -> Self;
    fn round(&self) -> Self;
    fn trunc(&self) -> Self;
    fn fract(&self) -> Self;
}

pub trait Fractional: Num
                    + Orderable
                    + Round
                    + Div<Self,Self> {
    fn recip(&self) -> Self;
}

pub trait Algebraic {
    fn pow(&self, n: &Self) -> Self;
    fn sqrt(&self) -> Self;
    fn rsqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn hypot(&self, other: &Self) -> Self;
}

/// Raise a number to a power.
///
/// # Example
///
/// ```rust
/// let sixteen: float = num::pow(2.0, 4.0);
/// assert_eq!(sixteen, 16.0);
/// ```
#[inline(always)] pub fn pow<T: Algebraic>(value: T, n: T) -> T { value.pow(&n) }
/// Take the squre root of a number.
#[inline(always)] pub fn sqrt<T: Algebraic>(value: T) -> T { value.sqrt() }
/// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
#[inline(always)] pub fn rsqrt<T: Algebraic>(value: T) -> T { value.rsqrt() }
/// Take the cubic root of a number.
#[inline(always)] pub fn cbrt<T: Algebraic>(value: T) -> T { value.cbrt() }
/// Calculate the length of the hypotenuse of a right-angle triangle given legs of length `x` and
/// `y`.
#[inline(always)] pub fn hypot<T: Algebraic>(x: T, y: T) -> T { x.hypot(&y) }

pub trait Trigonometric {
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;

    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;

    fn atan2(&self, other: &Self) -> Self;
    fn sin_cos(&self) -> (Self, Self);
}

/// Sine function.
#[inline(always)] pub fn sin<T: Trigonometric>(value: T) -> T { value.sin() }
/// Cosine function.
#[inline(always)] pub fn cos<T: Trigonometric>(value: T) -> T { value.cos() }
/// Tangent function.
#[inline(always)] pub fn tan<T: Trigonometric>(value: T) -> T { value.tan() }

/// Compute the arcsine of the number.
#[inline(always)] pub fn asin<T: Trigonometric>(value: T) -> T { value.asin() }
/// Compute the arccosine of the number.
#[inline(always)] pub fn acos<T: Trigonometric>(value: T) -> T { value.acos() }
/// Compute the arctangent of the number.
#[inline(always)] pub fn atan<T: Trigonometric>(value: T) -> T { value.atan() }

/// Compute the arctangent with 2 arguments.
#[inline(always)] pub fn atan2<T: Trigonometric>(x: T, y: T) -> T { x.atan2(&y) }
/// Simultaneously computes the sine and cosine of the number.
#[inline(always)] pub fn sin_cos<T: Trigonometric>(value: T) -> (T, T) { value.sin_cos() }

pub trait Exponential {
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;

    fn ln(&self) -> Self;
    fn log(&self, base: &Self) -> Self;
    fn log2(&self) -> Self;
    fn log10(&self) -> Self;
}

/// Returns `e^(value)`, (the exponential function).
#[inline(always)] pub fn exp<T: Exponential>(value: T) -> T { value.exp() }
/// Returns 2 raised to the power of the number, `2^(value)`.
#[inline(always)] pub fn exp2<T: Exponential>(value: T) -> T { value.exp2() }

/// Returns the natural logarithm of the number.
#[inline(always)] pub fn ln<T: Exponential>(value: T) -> T { value.ln() }
/// Returns the logarithm of the number with respect to an arbitrary base.
#[inline(always)] pub fn log<T: Exponential>(value: T, base: T) -> T { value.log(&base) }
/// Returns the base 2 logarithm of the number.
#[inline(always)] pub fn log2<T: Exponential>(value: T) -> T { value.log2() }
/// Returns the base 10 logarithm of the number.
#[inline(always)] pub fn log10<T: Exponential>(value: T) -> T { value.log10() }

pub trait Hyperbolic: Exponential {
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;

    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
}

/// Hyperbolic cosine function.
#[inline(always)] pub fn sinh<T: Hyperbolic>(value: T) -> T { value.sinh() }
/// Hyperbolic sine function.
#[inline(always)] pub fn cosh<T: Hyperbolic>(value: T) -> T { value.cosh() }
/// Hyperbolic tangent function.
#[inline(always)] pub fn tanh<T: Hyperbolic>(value: T) -> T { value.tanh() }

/// Inverse hyperbolic sine function.
#[inline(always)] pub fn asinh<T: Hyperbolic>(value: T) -> T { value.asinh() }
/// Inverse hyperbolic cosine function.
#[inline(always)] pub fn acosh<T: Hyperbolic>(value: T) -> T { value.acosh() }
/// Inverse hyperbolic tangent function.
#[inline(always)] pub fn atanh<T: Hyperbolic>(value: T) -> T { value.atanh() }

/// Defines constants and methods common to real numbers
pub trait Real: Signed
              + Fractional
              + Algebraic
              + Trigonometric
              + Hyperbolic {
    // Common Constants
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

    // Angular conversions
    fn to_degrees(&self) -> Self;
    fn to_radians(&self) -> Self;
}

/// Methods that are harder to implement and not commonly used.
pub trait RealExt: Real {
    // FIXME (#5527): usages of `int` should be replaced with an associated
    // integer type once these are implemented

    // Gamma functions
    fn lgamma(&self) -> (int, Self);
    fn tgamma(&self) -> Self;

    // Bessel functions
    fn j0(&self) -> Self;
    fn j1(&self) -> Self;
    fn jn(&self, n: int) -> Self;
    fn y0(&self) -> Self;
    fn y1(&self) -> Self;
    fn yn(&self, n: int) -> Self;
}

/// Collects the bitwise operators under one trait.
pub trait Bitwise: Not<Self>
                 + BitAnd<Self,Self>
                 + BitOr<Self,Self>
                 + BitXor<Self,Self>
                 + Shl<Self,Self>
                 + Shr<Self,Self> {}

pub trait BitCount {
    fn population_count(&self) -> Self;
    fn leading_zeros(&self) -> Self;
    fn trailing_zeros(&self) -> Self;
}

pub trait Bounded {
    // FIXME (#5527): These should be associated constants
    fn min_value() -> Self;
    fn max_value() -> Self;
}

/// Specifies the available operations common to all of Rust's core numeric primitives.
/// These may not always make sense from a purely mathematical point of view, but
/// may be useful for systems programming.
pub trait Primitive: Clone
                   + DeepClone
                   + Num
                   + NumCast
                   + Orderable
                   + Bounded
                   + Neg<Self>
                   + Add<Self,Self>
                   + Sub<Self,Self>
                   + Mul<Self,Self>
                   + Div<Self,Self>
                   + Rem<Self,Self> {
    // FIXME (#5527): These should be associated constants
    // FIXME (#8888): Removing `unused_self` requires #8888 to be fixed.
    fn bits(unused_self: Option<Self>) -> uint;
    fn bytes(unused_self: Option<Self>) -> uint;
    fn is_signed(unused_self: Option<Self>) -> bool;
}

/// A collection of traits relevant to primitive signed and unsigned integers
pub trait Int: Integer
             + Primitive
             + Bitwise
             + BitCount {}

/// Used for representing the classification of floating point numbers
#[deriving(Eq)]
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
pub trait Float: Real
               + Signed
               + Primitive
               + ApproxEq<Self> {
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
            if Primitive::bits(None::<$SrcT>) <= Primitive::bits(None::<$DstT>) {
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
            if Primitive::bits(None::<$SrcT>) <= Primitive::bits(None::<$DstT>) {
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
        if Primitive::bits(None::<$SrcT>) <= Primitive::bits(None::<$DstT>) {
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

/// Calculates a power to a given radix, optimized for uint `pow` and `radix`.
///
/// Returns `radix^pow` as `T`.
///
/// Note:
/// Also returns `1` for `0^0`, despite that technically being an
/// undefined number. The reason for this is twofold:
/// - If code written to use this function cares about that special case, it's
///   probably going to catch it before making the call.
/// - If code written to use this function doesn't care about it, it's
///   probably assuming that `x^0` always equals `1`.
///
pub fn pow_with_uint<T:NumCast+One+Zero+Div<T,T>+Mul<T,T>>(radix: uint, pow: uint) -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();

    if pow   == 0u { return _1; }
    if radix == 0u { return _0; }
    let mut my_pow     = pow;
    let mut total      = _1;
    let mut multiplier = cast(radix).unwrap();
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total = total * multiplier;
        }
        my_pow = my_pow / 2u;
        multiplier = multiplier * multiplier;
    }
    total
}

impl<T: Zero + 'static> Zero for @mut T {
    fn zero() -> @mut T { @mut Zero::zero() }
    fn is_zero(&self) -> bool { (**self).is_zero() }
}

impl<T: Zero + 'static> Zero for @T {
    fn zero() -> @T { @Zero::zero() }
    fn is_zero(&self) -> bool { (**self).is_zero() }
}

impl<T: Zero> Zero for ~T {
    fn zero() -> ~T { ~Zero::zero() }
    fn is_zero(&self) -> bool { (**self).is_zero() }
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
pub fn test_num<T:Num + NumCast>(ten: T, two: T) {
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
        assert_eq!(int::min_value.to_int(),  Some(int::min_value as int));
        assert_eq!(int::min_value.to_i8(),   None);
        assert_eq!(int::min_value.to_i16(),  None);
        // int::min_value.to_i32() is word-size specific
        assert_eq!(int::min_value.to_i64(),  Some(int::min_value as i64));
        assert_eq!(int::min_value.to_uint(), None);
        assert_eq!(int::min_value.to_u8(),   None);
        assert_eq!(int::min_value.to_u16(),  None);
        assert_eq!(int::min_value.to_u32(),  None);
        assert_eq!(int::min_value.to_u64(),  None);

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(int::min_value.to_i32(), Some(int::min_value as i32));
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(int::min_value.to_i32(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_i8_min() {
        assert_eq!(i8::min_value.to_int(),  Some(i8::min_value as int));
        assert_eq!(i8::min_value.to_i8(),   Some(i8::min_value as i8));
        assert_eq!(i8::min_value.to_i16(),  Some(i8::min_value as i16));
        assert_eq!(i8::min_value.to_i32(),  Some(i8::min_value as i32));
        assert_eq!(i8::min_value.to_i64(),  Some(i8::min_value as i64));
        assert_eq!(i8::min_value.to_uint(), None);
        assert_eq!(i8::min_value.to_u8(),   None);
        assert_eq!(i8::min_value.to_u16(),  None);
        assert_eq!(i8::min_value.to_u32(),  None);
        assert_eq!(i8::min_value.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i16_min() {
        assert_eq!(i16::min_value.to_int(),  Some(i16::min_value as int));
        assert_eq!(i16::min_value.to_i8(),   None);
        assert_eq!(i16::min_value.to_i16(),  Some(i16::min_value as i16));
        assert_eq!(i16::min_value.to_i32(),  Some(i16::min_value as i32));
        assert_eq!(i16::min_value.to_i64(),  Some(i16::min_value as i64));
        assert_eq!(i16::min_value.to_uint(), None);
        assert_eq!(i16::min_value.to_u8(),   None);
        assert_eq!(i16::min_value.to_u16(),  None);
        assert_eq!(i16::min_value.to_u32(),  None);
        assert_eq!(i16::min_value.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i32_min() {
        assert_eq!(i32::min_value.to_int(),  Some(i32::min_value as int));
        assert_eq!(i32::min_value.to_i8(),   None);
        assert_eq!(i32::min_value.to_i16(),  None);
        assert_eq!(i32::min_value.to_i32(),  Some(i32::min_value as i32));
        assert_eq!(i32::min_value.to_i64(),  Some(i32::min_value as i64));
        assert_eq!(i32::min_value.to_uint(), None);
        assert_eq!(i32::min_value.to_u8(),   None);
        assert_eq!(i32::min_value.to_u16(),  None);
        assert_eq!(i32::min_value.to_u32(),  None);
        assert_eq!(i32::min_value.to_u64(),  None);
    }

    #[test]
    fn test_cast_range_i64_min() {
        // i64::min_value.to_int() is word-size specific
        assert_eq!(i64::min_value.to_i8(),   None);
        assert_eq!(i64::min_value.to_i16(),  None);
        assert_eq!(i64::min_value.to_i32(),  None);
        assert_eq!(i64::min_value.to_i64(),  Some(i64::min_value as i64));
        assert_eq!(i64::min_value.to_uint(), None);
        assert_eq!(i64::min_value.to_u8(),   None);
        assert_eq!(i64::min_value.to_u16(),  None);
        assert_eq!(i64::min_value.to_u32(),  None);
        assert_eq!(i64::min_value.to_u64(),  None);

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(i64::min_value.to_int(), None);
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(i64::min_value.to_int(), Some(i64::min_value as int));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_int_max() {
        assert_eq!(int::max_value.to_int(),  Some(int::max_value as int));
        assert_eq!(int::max_value.to_i8(),   None);
        assert_eq!(int::max_value.to_i16(),  None);
        // int::max_value.to_i32() is word-size specific
        assert_eq!(int::max_value.to_i64(),  Some(int::max_value as i64));
        assert_eq!(int::max_value.to_u8(),   None);
        assert_eq!(int::max_value.to_u16(),  None);
        // int::max_value.to_u32() is word-size specific
        assert_eq!(int::max_value.to_u64(),  Some(int::max_value as u64));

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(int::max_value.to_i32(), Some(int::max_value as i32));
            assert_eq!(int::max_value.to_u32(), Some(int::max_value as u32));
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(int::max_value.to_i32(), None);
            assert_eq!(int::max_value.to_u32(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_i8_max() {
        assert_eq!(i8::max_value.to_int(),  Some(i8::max_value as int));
        assert_eq!(i8::max_value.to_i8(),   Some(i8::max_value as i8));
        assert_eq!(i8::max_value.to_i16(),  Some(i8::max_value as i16));
        assert_eq!(i8::max_value.to_i32(),  Some(i8::max_value as i32));
        assert_eq!(i8::max_value.to_i64(),  Some(i8::max_value as i64));
        assert_eq!(i8::max_value.to_uint(), Some(i8::max_value as uint));
        assert_eq!(i8::max_value.to_u8(),   Some(i8::max_value as u8));
        assert_eq!(i8::max_value.to_u16(),  Some(i8::max_value as u16));
        assert_eq!(i8::max_value.to_u32(),  Some(i8::max_value as u32));
        assert_eq!(i8::max_value.to_u64(),  Some(i8::max_value as u64));
    }

    #[test]
    fn test_cast_range_i16_max() {
        assert_eq!(i16::max_value.to_int(),  Some(i16::max_value as int));
        assert_eq!(i16::max_value.to_i8(),   None);
        assert_eq!(i16::max_value.to_i16(),  Some(i16::max_value as i16));
        assert_eq!(i16::max_value.to_i32(),  Some(i16::max_value as i32));
        assert_eq!(i16::max_value.to_i64(),  Some(i16::max_value as i64));
        assert_eq!(i16::max_value.to_uint(), Some(i16::max_value as uint));
        assert_eq!(i16::max_value.to_u8(),   None);
        assert_eq!(i16::max_value.to_u16(),  Some(i16::max_value as u16));
        assert_eq!(i16::max_value.to_u32(),  Some(i16::max_value as u32));
        assert_eq!(i16::max_value.to_u64(),  Some(i16::max_value as u64));
    }

    #[test]
    fn test_cast_range_i32_max() {
        assert_eq!(i32::max_value.to_int(),  Some(i32::max_value as int));
        assert_eq!(i32::max_value.to_i8(),   None);
        assert_eq!(i32::max_value.to_i16(),  None);
        assert_eq!(i32::max_value.to_i32(),  Some(i32::max_value as i32));
        assert_eq!(i32::max_value.to_i64(),  Some(i32::max_value as i64));
        assert_eq!(i32::max_value.to_uint(), Some(i32::max_value as uint));
        assert_eq!(i32::max_value.to_u8(),   None);
        assert_eq!(i32::max_value.to_u16(),  None);
        assert_eq!(i32::max_value.to_u32(),  Some(i32::max_value as u32));
        assert_eq!(i32::max_value.to_u64(),  Some(i32::max_value as u64));
    }

    #[test]
    fn test_cast_range_i64_max() {
        // i64::max_value.to_int() is word-size specific
        assert_eq!(i64::max_value.to_i8(),   None);
        assert_eq!(i64::max_value.to_i16(),  None);
        assert_eq!(i64::max_value.to_i32(),  None);
        assert_eq!(i64::max_value.to_i64(),  Some(i64::max_value as i64));
        // i64::max_value.to_uint() is word-size specific
        assert_eq!(i64::max_value.to_u8(),   None);
        assert_eq!(i64::max_value.to_u16(),  None);
        assert_eq!(i64::max_value.to_u32(),  None);
        assert_eq!(i64::max_value.to_u64(),  Some(i64::max_value as u64));

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(i64::max_value.to_int(),  None);
            assert_eq!(i64::max_value.to_uint(), None);
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(i64::max_value.to_int(),  Some(i64::max_value as int));
            assert_eq!(i64::max_value.to_uint(), Some(i64::max_value as uint));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_uint_min() {
        assert_eq!(uint::min_value.to_int(),  Some(uint::min_value as int));
        assert_eq!(uint::min_value.to_i8(),   Some(uint::min_value as i8));
        assert_eq!(uint::min_value.to_i16(),  Some(uint::min_value as i16));
        assert_eq!(uint::min_value.to_i32(),  Some(uint::min_value as i32));
        assert_eq!(uint::min_value.to_i64(),  Some(uint::min_value as i64));
        assert_eq!(uint::min_value.to_uint(), Some(uint::min_value as uint));
        assert_eq!(uint::min_value.to_u8(),   Some(uint::min_value as u8));
        assert_eq!(uint::min_value.to_u16(),  Some(uint::min_value as u16));
        assert_eq!(uint::min_value.to_u32(),  Some(uint::min_value as u32));
        assert_eq!(uint::min_value.to_u64(),  Some(uint::min_value as u64));
    }

    #[test]
    fn test_cast_range_u8_min() {
        assert_eq!(u8::min_value.to_int(),  Some(u8::min_value as int));
        assert_eq!(u8::min_value.to_i8(),   Some(u8::min_value as i8));
        assert_eq!(u8::min_value.to_i16(),  Some(u8::min_value as i16));
        assert_eq!(u8::min_value.to_i32(),  Some(u8::min_value as i32));
        assert_eq!(u8::min_value.to_i64(),  Some(u8::min_value as i64));
        assert_eq!(u8::min_value.to_uint(), Some(u8::min_value as uint));
        assert_eq!(u8::min_value.to_u8(),   Some(u8::min_value as u8));
        assert_eq!(u8::min_value.to_u16(),  Some(u8::min_value as u16));
        assert_eq!(u8::min_value.to_u32(),  Some(u8::min_value as u32));
        assert_eq!(u8::min_value.to_u64(),  Some(u8::min_value as u64));
    }

    #[test]
    fn test_cast_range_u16_min() {
        assert_eq!(u16::min_value.to_int(),  Some(u16::min_value as int));
        assert_eq!(u16::min_value.to_i8(),   Some(u16::min_value as i8));
        assert_eq!(u16::min_value.to_i16(),  Some(u16::min_value as i16));
        assert_eq!(u16::min_value.to_i32(),  Some(u16::min_value as i32));
        assert_eq!(u16::min_value.to_i64(),  Some(u16::min_value as i64));
        assert_eq!(u16::min_value.to_uint(), Some(u16::min_value as uint));
        assert_eq!(u16::min_value.to_u8(),   Some(u16::min_value as u8));
        assert_eq!(u16::min_value.to_u16(),  Some(u16::min_value as u16));
        assert_eq!(u16::min_value.to_u32(),  Some(u16::min_value as u32));
        assert_eq!(u16::min_value.to_u64(),  Some(u16::min_value as u64));
    }

    #[test]
    fn test_cast_range_u32_min() {
        assert_eq!(u32::min_value.to_int(),  Some(u32::min_value as int));
        assert_eq!(u32::min_value.to_i8(),   Some(u32::min_value as i8));
        assert_eq!(u32::min_value.to_i16(),  Some(u32::min_value as i16));
        assert_eq!(u32::min_value.to_i32(),  Some(u32::min_value as i32));
        assert_eq!(u32::min_value.to_i64(),  Some(u32::min_value as i64));
        assert_eq!(u32::min_value.to_uint(), Some(u32::min_value as uint));
        assert_eq!(u32::min_value.to_u8(),   Some(u32::min_value as u8));
        assert_eq!(u32::min_value.to_u16(),  Some(u32::min_value as u16));
        assert_eq!(u32::min_value.to_u32(),  Some(u32::min_value as u32));
        assert_eq!(u32::min_value.to_u64(),  Some(u32::min_value as u64));
    }

    #[test]
    fn test_cast_range_u64_min() {
        assert_eq!(u64::min_value.to_int(),  Some(u64::min_value as int));
        assert_eq!(u64::min_value.to_i8(),   Some(u64::min_value as i8));
        assert_eq!(u64::min_value.to_i16(),  Some(u64::min_value as i16));
        assert_eq!(u64::min_value.to_i32(),  Some(u64::min_value as i32));
        assert_eq!(u64::min_value.to_i64(),  Some(u64::min_value as i64));
        assert_eq!(u64::min_value.to_uint(), Some(u64::min_value as uint));
        assert_eq!(u64::min_value.to_u8(),   Some(u64::min_value as u8));
        assert_eq!(u64::min_value.to_u16(),  Some(u64::min_value as u16));
        assert_eq!(u64::min_value.to_u32(),  Some(u64::min_value as u32));
        assert_eq!(u64::min_value.to_u64(),  Some(u64::min_value as u64));
    }

    #[test]
    fn test_cast_range_uint_max() {
        assert_eq!(uint::max_value.to_int(),  None);
        assert_eq!(uint::max_value.to_i8(),   None);
        assert_eq!(uint::max_value.to_i16(),  None);
        assert_eq!(uint::max_value.to_i32(),  None);
        // uint::max_value.to_i64() is word-size specific
        assert_eq!(uint::max_value.to_u8(),   None);
        assert_eq!(uint::max_value.to_u16(),  None);
        // uint::max_value.to_u32() is word-size specific
        assert_eq!(uint::max_value.to_u64(),  Some(uint::max_value as u64));

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(uint::max_value.to_u32(), Some(uint::max_value as u32));
            assert_eq!(uint::max_value.to_i64(), Some(uint::max_value as i64));
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(uint::max_value.to_u32(), None);
            assert_eq!(uint::max_value.to_i64(), None);
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_u8_max() {
        assert_eq!(u8::max_value.to_int(),  Some(u8::max_value as int));
        assert_eq!(u8::max_value.to_i8(),   None);
        assert_eq!(u8::max_value.to_i16(),  Some(u8::max_value as i16));
        assert_eq!(u8::max_value.to_i32(),  Some(u8::max_value as i32));
        assert_eq!(u8::max_value.to_i64(),  Some(u8::max_value as i64));
        assert_eq!(u8::max_value.to_uint(), Some(u8::max_value as uint));
        assert_eq!(u8::max_value.to_u8(),   Some(u8::max_value as u8));
        assert_eq!(u8::max_value.to_u16(),  Some(u8::max_value as u16));
        assert_eq!(u8::max_value.to_u32(),  Some(u8::max_value as u32));
        assert_eq!(u8::max_value.to_u64(),  Some(u8::max_value as u64));
    }

    #[test]
    fn test_cast_range_u16_max() {
        assert_eq!(u16::max_value.to_int(),  Some(u16::max_value as int));
        assert_eq!(u16::max_value.to_i8(),   None);
        assert_eq!(u16::max_value.to_i16(),  None);
        assert_eq!(u16::max_value.to_i32(),  Some(u16::max_value as i32));
        assert_eq!(u16::max_value.to_i64(),  Some(u16::max_value as i64));
        assert_eq!(u16::max_value.to_uint(), Some(u16::max_value as uint));
        assert_eq!(u16::max_value.to_u8(),   None);
        assert_eq!(u16::max_value.to_u16(),  Some(u16::max_value as u16));
        assert_eq!(u16::max_value.to_u32(),  Some(u16::max_value as u32));
        assert_eq!(u16::max_value.to_u64(),  Some(u16::max_value as u64));
    }

    #[test]
    fn test_cast_range_u32_max() {
        // u32::max_value.to_int() is word-size specific
        assert_eq!(u32::max_value.to_i8(),   None);
        assert_eq!(u32::max_value.to_i16(),  None);
        assert_eq!(u32::max_value.to_i32(),  None);
        assert_eq!(u32::max_value.to_i64(),  Some(u32::max_value as i64));
        assert_eq!(u32::max_value.to_uint(), Some(u32::max_value as uint));
        assert_eq!(u32::max_value.to_u8(),   None);
        assert_eq!(u32::max_value.to_u16(),  None);
        assert_eq!(u32::max_value.to_u32(),  Some(u32::max_value as u32));
        assert_eq!(u32::max_value.to_u64(),  Some(u32::max_value as u64));

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(u32::max_value.to_int(),  None);
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(u32::max_value.to_int(),  Some(u32::max_value as int));
        }

        check_word_size();
    }

    #[test]
    fn test_cast_range_u64_max() {
        assert_eq!(u64::max_value.to_int(),  None);
        assert_eq!(u64::max_value.to_i8(),   None);
        assert_eq!(u64::max_value.to_i16(),  None);
        assert_eq!(u64::max_value.to_i32(),  None);
        assert_eq!(u64::max_value.to_i64(),  None);
        // u64::max_value.to_uint() is word-size specific
        assert_eq!(u64::max_value.to_u8(),   None);
        assert_eq!(u64::max_value.to_u16(),  None);
        assert_eq!(u64::max_value.to_u32(),  None);
        assert_eq!(u64::max_value.to_u64(),  Some(u64::max_value as u64));

        #[cfg(target_word_size = "32")]
        fn check_word_size() {
            assert_eq!(u64::max_value.to_uint(), None);
        }

        #[cfg(target_word_size = "64")]
        fn check_word_size() {
            assert_eq!(u64::max_value.to_uint(), Some(u64::max_value as uint));
        }

        check_word_size();
    }

    #[test]
    fn test_saturating_add_uint() {
        use uint::max_value;
        assert_eq!(3u.saturating_add(5u), 8u);
        assert_eq!(3u.saturating_add(max_value-1), max_value);
        assert_eq!(max_value.saturating_add(max_value), max_value);
        assert_eq!((max_value-2).saturating_add(1), max_value-1);
    }

    #[test]
    fn test_saturating_sub_uint() {
        use uint::max_value;
        assert_eq!(5u.saturating_sub(3u), 2u);
        assert_eq!(3u.saturating_sub(5u), 0u);
        assert_eq!(0u.saturating_sub(1u), 0u);
        assert_eq!((max_value-1).saturating_sub(max_value), 0);
    }

    #[test]
    fn test_saturating_add_int() {
        use int::{min_value,max_value};
        assert_eq!(3i.saturating_add(5i), 8i);
        assert_eq!(3i.saturating_add(max_value-1), max_value);
        assert_eq!(max_value.saturating_add(max_value), max_value);
        assert_eq!((max_value-2).saturating_add(1), max_value-1);
        assert_eq!(3i.saturating_add(-5i), -2i);
        assert_eq!(min_value.saturating_add(-1i), min_value);
        assert_eq!((-2i).saturating_add(-max_value), min_value);
    }

    #[test]
    fn test_saturating_sub_int() {
        use int::{min_value,max_value};
        assert_eq!(3i.saturating_sub(5i), -2i);
        assert_eq!(min_value.saturating_sub(1i), min_value);
        assert_eq!((-2i).saturating_sub(max_value), min_value);
        assert_eq!(3i.saturating_sub(-5i), 8i);
        assert_eq!(3i.saturating_sub(-(max_value-1)), max_value);
        assert_eq!(max_value.saturating_sub(-max_value), max_value);
        assert_eq!((max_value-2).saturating_sub(-1), max_value-1);
    }

    #[test]
    fn test_checked_add() {
        let five_less = uint::max_value - 5;
        assert_eq!(five_less.checked_add(&0), Some(uint::max_value - 5));
        assert_eq!(five_less.checked_add(&1), Some(uint::max_value - 4));
        assert_eq!(five_less.checked_add(&2), Some(uint::max_value - 3));
        assert_eq!(five_less.checked_add(&3), Some(uint::max_value - 2));
        assert_eq!(five_less.checked_add(&4), Some(uint::max_value - 1));
        assert_eq!(five_less.checked_add(&5), Some(uint::max_value));
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
        let third = uint::max_value / 3;
        assert_eq!(third.checked_mul(&0), Some(0));
        assert_eq!(third.checked_mul(&1), Some(third));
        assert_eq!(third.checked_mul(&2), Some(third * 2));
        assert_eq!(third.checked_mul(&3), Some(third * 3));
        assert_eq!(third.checked_mul(&4), None);
    }


    #[deriving(Eq)]
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
}

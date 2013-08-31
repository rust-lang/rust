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

use cmp::{Eq, ApproxEq, Ord};
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use ops::{Not, BitAnd, BitOr, BitXor, Shl, Shr};
use option::{Option, Some, None};
use unstable::intrinsics;

pub mod strconv;

/// The base trait for numeric types
pub trait Num: Eq + Zero + One
             + Neg<Self>
             + Add<Self,Self>
             + Sub<Self,Self>
             + Mul<Self,Self>
             + Div<Self,Self>
             + Rem<Self,Self> {}

pub trait IntConvertible {
    fn to_int(&self) -> int;
    fn from_int(n: int) -> Self;
}

pub trait Orderable: Ord {
    // These should be methods on `Ord`, with overridable default implementations. We don't want
    // to encumber all implementors of Ord by requiring them to implement these functions, but at
    // the same time we want to be able to take advantage of the speed of the specific numeric
    // functions (like the `fmin` and `fmax` intrinsics).
    fn min(&self, other: &Self) -> Self;
    fn max(&self, other: &Self) -> Self;
    fn clamp(&self, mn: &Self, mx: &Self) -> Self;
}

#[inline(always)] pub fn min<T: Orderable>(x: T, y: T) -> T { x.min(&y) }
#[inline(always)] pub fn max<T: Orderable>(x: T, y: T) -> T { x.max(&y) }
#[inline(always)] pub fn clamp<T: Orderable>(value: T, mn: T, mx: T) -> T { value.clamp(&mn, &mx) }

pub trait Zero {
    fn zero() -> Self;      // FIXME (#5527): This should be an associated constant
    fn is_zero(&self) -> bool;
}

#[inline(always)] pub fn zero<T: Zero>() -> T { Zero::zero() }

pub trait One {
    fn one() -> Self;       // FIXME (#5527): This should be an associated constant
}

#[inline(always)] pub fn one<T: One>() -> T { One::one() }

pub trait Signed: Num
                + Neg<Self> {
    fn abs(&self) -> Self;
    fn abs_sub(&self, other: &Self) -> Self;
    fn signum(&self) -> Self;

    fn is_positive(&self) -> bool;
    fn is_negative(&self) -> bool;
}

#[inline(always)] pub fn abs<T: Signed>(value: T) -> T { value.abs() }
#[inline(always)] pub fn abs_sub<T: Signed>(x: T, y: T) -> T { x.abs_sub(&y) }
#[inline(always)] pub fn signum<T: Signed>(value: T) -> T { value.signum() }

pub trait Unsigned: Num {}

/// Times trait
///
/// ~~~ {.rust}
/// use num::Times;
/// let ten = 10 as uint;
/// let mut accum = 0;
/// do ten.times { accum += 1; }
/// ~~~
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

#[inline(always)] pub fn gcd<T: Integer>(x: T, y: T) -> T { x.gcd(&y) }
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

#[inline(always)] pub fn pow<T: Algebraic>(value: T, n: T) -> T { value.pow(&n) }
#[inline(always)] pub fn sqrt<T: Algebraic>(value: T) -> T { value.sqrt() }
#[inline(always)] pub fn rsqrt<T: Algebraic>(value: T) -> T { value.rsqrt() }
#[inline(always)] pub fn cbrt<T: Algebraic>(value: T) -> T { value.cbrt() }
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

#[inline(always)] pub fn sin<T: Trigonometric>(value: T) -> T { value.sin() }
#[inline(always)] pub fn cos<T: Trigonometric>(value: T) -> T { value.cos() }
#[inline(always)] pub fn tan<T: Trigonometric>(value: T) -> T { value.tan() }

#[inline(always)] pub fn asin<T: Trigonometric>(value: T) -> T { value.asin() }
#[inline(always)] pub fn acos<T: Trigonometric>(value: T) -> T { value.acos() }
#[inline(always)] pub fn atan<T: Trigonometric>(value: T) -> T { value.atan() }

#[inline(always)] pub fn atan2<T: Trigonometric>(x: T, y: T) -> T { x.atan2(&y) }
#[inline(always)] pub fn sin_cos<T: Trigonometric>(value: T) -> (T, T) { value.sin_cos() }

pub trait Exponential {
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;

    fn ln(&self) -> Self;
    fn log(&self, base: &Self) -> Self;
    fn log2(&self) -> Self;
    fn log10(&self) -> Self;
}

#[inline(always)] pub fn exp<T: Exponential>(value: T) -> T { value.exp() }
#[inline(always)] pub fn exp2<T: Exponential>(value: T) -> T { value.exp2() }

#[inline(always)] pub fn ln<T: Exponential>(value: T) -> T { value.ln() }
#[inline(always)] pub fn log<T: Exponential>(value: T, base: T) -> T { value.log(&base) }
#[inline(always)] pub fn log2<T: Exponential>(value: T) -> T { value.log2() }
#[inline(always)] pub fn log10<T: Exponential>(value: T) -> T { value.log10() }

pub trait Hyperbolic: Exponential {
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;

    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
}

#[inline(always)] pub fn sinh<T: Hyperbolic>(value: T) -> T { value.sinh() }
#[inline(always)] pub fn cosh<T: Hyperbolic>(value: T) -> T { value.cosh() }
#[inline(always)] pub fn tanh<T: Hyperbolic>(value: T) -> T { value.tanh() }

#[inline(always)] pub fn asinh<T: Hyperbolic>(value: T) -> T { value.asinh() }
#[inline(always)] pub fn acosh<T: Hyperbolic>(value: T) -> T { value.acosh() }
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
pub trait Primitive: Num
                   + NumCast
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
    fn NaN() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn neg_zero() -> Self;

    fn is_NaN(&self) -> bool;
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

#[inline(always)] pub fn exp_m1<T: Float>(value: T) -> T { value.exp_m1() }
#[inline(always)] pub fn ln_1p<T: Float>(value: T) -> T { value.ln_1p() }
#[inline(always)] pub fn mul_add<T: Float>(a: T, b: T, c: T) -> T { a.mul_add(b, c) }

/// Cast from one machine scalar to another
///
/// # Example
///
/// ~~~
/// let twenty: f32 = num::cast(0x14);
/// assert_eq!(twenty, 20f32);
/// ~~~
///
#[inline]
pub fn cast<T:NumCast,U:NumCast>(n: T) -> U {
    NumCast::from(n)
}

/// An interface for casting between machine scalars
pub trait NumCast {
    fn from<T:NumCast>(n: T) -> Self;

    fn to_u8(&self) -> u8;
    fn to_u16(&self) -> u16;
    fn to_u32(&self) -> u32;
    fn to_u64(&self) -> u64;
    fn to_uint(&self) -> uint;

    fn to_i8(&self) -> i8;
    fn to_i16(&self) -> i16;
    fn to_i32(&self) -> i32;
    fn to_i64(&self) -> i64;
    fn to_int(&self) -> int;

    fn to_f32(&self) -> f32;
    fn to_f64(&self) -> f64;
    fn to_float(&self) -> float;
}

macro_rules! impl_num_cast(
    ($T:ty, $conv:ident) => (
        impl NumCast for $T {
            #[inline]
            fn from<N:NumCast>(n: N) -> $T {
                // `$conv` could be generated using `concat_idents!`, but that
                // macro seems to be broken at the moment
                n.$conv()
            }

            #[inline] fn to_u8(&self)    -> u8    { *self as u8    }
            #[inline] fn to_u16(&self)   -> u16   { *self as u16   }
            #[inline] fn to_u32(&self)   -> u32   { *self as u32   }
            #[inline] fn to_u64(&self)   -> u64   { *self as u64   }
            #[inline] fn to_uint(&self)  -> uint  { *self as uint  }

            #[inline] fn to_i8(&self)    -> i8    { *self as i8    }
            #[inline] fn to_i16(&self)   -> i16   { *self as i16   }
            #[inline] fn to_i32(&self)   -> i32   { *self as i32   }
            #[inline] fn to_i64(&self)   -> i64   { *self as i64   }
            #[inline] fn to_int(&self)   -> int   { *self as int   }

            #[inline] fn to_f32(&self)   -> f32   { *self as f32   }
            #[inline] fn to_f64(&self)   -> f64   { *self as f64   }
            #[inline] fn to_float(&self) -> float { *self as float }
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
impl_num_cast!(float, to_float)

pub trait ToStrRadix {
    fn to_str_radix(&self, radix: uint) -> ~str;
}

pub trait FromStrRadix {
    fn from_str_radix(str: &str, radix: uint) -> Option<Self>;
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
    let mut multiplier = cast(radix);
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

impl<T: CheckedAdd+CheckedSub+Zero+Ord+Bounded> Saturating for T {
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

impl CheckedAdd for i8 {
    #[inline]
    fn checked_add(&self, v: &i8) -> Option<i8> {
        unsafe {
            let (x, y) = intrinsics::i8_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedAdd for i16 {
    #[inline]
    fn checked_add(&self, v: &i16) -> Option<i16> {
        unsafe {
            let (x, y) = intrinsics::i16_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedAdd for i32 {
    #[inline]
    fn checked_add(&self, v: &i32) -> Option<i32> {
        unsafe {
            let (x, y) = intrinsics::i32_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedAdd for i64 {
    #[inline]
    fn checked_add(&self, v: &i64) -> Option<i64> {
        unsafe {
            let (x, y) = intrinsics::i64_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedAdd for int {
    #[inline]
    fn checked_add(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i32_add_with_overflow(*self as i32, *v as i32);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedAdd for int {
    #[inline]
    fn checked_add(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i64_add_with_overflow(*self as i64, *v as i64);
            if y { None } else { Some(x as int) }
        }
    }
}

impl CheckedAdd for u8 {
    #[inline]
    fn checked_add(&self, v: &u8) -> Option<u8> {
        unsafe {
            let (x, y) = intrinsics::u8_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedAdd for u16 {
    #[inline]
    fn checked_add(&self, v: &u16) -> Option<u16> {
        unsafe {
            let (x, y) = intrinsics::u16_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedAdd for u32 {
    #[inline]
    fn checked_add(&self, v: &u32) -> Option<u32> {
        unsafe {
            let (x, y) = intrinsics::u32_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedAdd for u64 {
    #[inline]
    fn checked_add(&self, v: &u64) -> Option<u64> {
        unsafe {
            let (x, y) = intrinsics::u64_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedAdd for uint {
    #[inline]
    fn checked_add(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_add_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedAdd for uint {
    #[inline]
    fn checked_add(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_add_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

pub trait CheckedSub: Sub<Self, Self> {
    fn checked_sub(&self, v: &Self) -> Option<Self>;
}

impl CheckedSub for i8 {
    #[inline]
    fn checked_sub(&self, v: &i8) -> Option<i8> {
        unsafe {
            let (x, y) = intrinsics::i8_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for i16 {
    #[inline]
    fn checked_sub(&self, v: &i16) -> Option<i16> {
        unsafe {
            let (x, y) = intrinsics::i16_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for i32 {
    #[inline]
    fn checked_sub(&self, v: &i32) -> Option<i32> {
        unsafe {
            let (x, y) = intrinsics::i32_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for i64 {
    #[inline]
    fn checked_sub(&self, v: &i64) -> Option<i64> {
        unsafe {
            let (x, y) = intrinsics::i64_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedSub for int {
    #[inline]
    fn checked_sub(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i32_sub_with_overflow(*self as i32, *v as i32);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedSub for int {
    #[inline]
    fn checked_sub(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i64_sub_with_overflow(*self as i64, *v as i64);
            if y { None } else { Some(x as int) }
        }
    }
}

impl CheckedSub for u8 {
    #[inline]
    fn checked_sub(&self, v: &u8) -> Option<u8> {
        unsafe {
            let (x, y) = intrinsics::u8_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for u16 {
    #[inline]
    fn checked_sub(&self, v: &u16) -> Option<u16> {
        unsafe {
            let (x, y) = intrinsics::u16_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for u32 {
    #[inline]
    fn checked_sub(&self, v: &u32) -> Option<u32> {
        unsafe {
            let (x, y) = intrinsics::u32_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for u64 {
    #[inline]
    fn checked_sub(&self, v: &u64) -> Option<u64> {
        unsafe {
            let (x, y) = intrinsics::u64_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedSub for uint {
    #[inline]
    fn checked_sub(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_sub_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedSub for uint {
    #[inline]
    fn checked_sub(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_sub_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

pub trait CheckedMul: Mul<Self, Self> {
    fn checked_mul(&self, v: &Self) -> Option<Self>;
}

impl CheckedMul for i8 {
    #[inline]
    fn checked_mul(&self, v: &i8) -> Option<i8> {
        unsafe {
            let (x, y) = intrinsics::i8_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for i16 {
    #[inline]
    fn checked_mul(&self, v: &i16) -> Option<i16> {
        unsafe {
            let (x, y) = intrinsics::i16_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for i32 {
    #[inline]
    fn checked_mul(&self, v: &i32) -> Option<i32> {
        unsafe {
            let (x, y) = intrinsics::i32_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

// FIXME: #8449: should not be disabled on 32-bit
#[cfg(target_word_size = "64")]
impl CheckedMul for i64 {
    #[inline]
    fn checked_mul(&self, v: &i64) -> Option<i64> {
        unsafe {
            let (x, y) = intrinsics::i64_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedMul for int {
    #[inline]
    fn checked_mul(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i32_mul_with_overflow(*self as i32, *v as i32);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedMul for int {
    #[inline]
    fn checked_mul(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i64_mul_with_overflow(*self as i64, *v as i64);
            if y { None } else { Some(x as int) }
        }
    }
}

impl CheckedMul for u8 {
    #[inline]
    fn checked_mul(&self, v: &u8) -> Option<u8> {
        unsafe {
            let (x, y) = intrinsics::u8_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for u16 {
    #[inline]
    fn checked_mul(&self, v: &u16) -> Option<u16> {
        unsafe {
            let (x, y) = intrinsics::u16_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for u32 {
    #[inline]
    fn checked_mul(&self, v: &u32) -> Option<u32> {
        unsafe {
            let (x, y) = intrinsics::u32_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

// FIXME: #8449: should not be disabled on 32-bit
#[cfg(target_word_size = "64")]
impl CheckedMul for u64 {
    #[inline]
    fn checked_mul(&self, v: &u64) -> Option<u64> {
        unsafe {
            let (x, y) = intrinsics::u64_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedMul for uint {
    #[inline]
    fn checked_mul(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_mul_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedMul for uint {
    #[inline]
    fn checked_mul(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_mul_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

pub trait CheckedDiv: Div<Self, Self> {
    fn checked_div(&self, v: &Self) -> Option<Self>;
}

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T:Num + NumCast>(ten: T, two: T) {
    assert_eq!(ten.add(&two),  cast(12));
    assert_eq!(ten.sub(&two),  cast(8));
    assert_eq!(ten.mul(&two),  cast(20));
    assert_eq!(ten.div(&two),  cast(5));
    assert_eq!(ten.rem(&two),  cast(0));

    assert_eq!(ten.add(&two),  ten + two);
    assert_eq!(ten.sub(&two),  ten - two);
    assert_eq!(ten.mul(&two),  ten * two);
    assert_eq!(ten.div(&two),  ten / two);
    assert_eq!(ten.rem(&two),  ten % two);
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use uint;
    use super::*;

    macro_rules! test_cast_20(
        ($_20:expr) => ({
            let _20 = $_20;

            assert_eq!(20u,   _20.to_uint());
            assert_eq!(20u8,  _20.to_u8());
            assert_eq!(20u16, _20.to_u16());
            assert_eq!(20u32, _20.to_u32());
            assert_eq!(20u64, _20.to_u64());
            assert_eq!(20i,   _20.to_int());
            assert_eq!(20i8,  _20.to_i8());
            assert_eq!(20i16, _20.to_i16());
            assert_eq!(20i32, _20.to_i32());
            assert_eq!(20i64, _20.to_i64());
            assert_eq!(20f,   _20.to_float());
            assert_eq!(20f32, _20.to_f32());
            assert_eq!(20f64, _20.to_f64());

            assert_eq!(_20, NumCast::from(20u));
            assert_eq!(_20, NumCast::from(20u8));
            assert_eq!(_20, NumCast::from(20u16));
            assert_eq!(_20, NumCast::from(20u32));
            assert_eq!(_20, NumCast::from(20u64));
            assert_eq!(_20, NumCast::from(20i));
            assert_eq!(_20, NumCast::from(20i8));
            assert_eq!(_20, NumCast::from(20i16));
            assert_eq!(_20, NumCast::from(20i32));
            assert_eq!(_20, NumCast::from(20i64));
            assert_eq!(_20, NumCast::from(20f));
            assert_eq!(_20, NumCast::from(20f32));
            assert_eq!(_20, NumCast::from(20f64));

            assert_eq!(_20, cast(20u));
            assert_eq!(_20, cast(20u8));
            assert_eq!(_20, cast(20u16));
            assert_eq!(_20, cast(20u32));
            assert_eq!(_20, cast(20u64));
            assert_eq!(_20, cast(20i));
            assert_eq!(_20, cast(20i8));
            assert_eq!(_20, cast(20i16));
            assert_eq!(_20, cast(20i32));
            assert_eq!(_20, cast(20i64));
            assert_eq!(_20, cast(20f));
            assert_eq!(_20, cast(20f32));
            assert_eq!(_20, cast(20f64));
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
    #[test] fn test_float_cast() { test_cast_20!(20f)   }

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
}

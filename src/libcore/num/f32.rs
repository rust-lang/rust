// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for 32-bits floats (`f32` type)

use default::Default;
use intrinsics;
use num::{Zero, One, Bounded, Signed, Num, Primitive};

#[cfg(not(test))] use cmp::{Eq, Ord};
#[cfg(not(test))] use ops::{Add, Sub, Mul, Div, Rem, Neg};

pub static RADIX: uint = 2u;

pub static MANTISSA_DIGITS: uint = 24u;
pub static DIGITS: uint = 6u;

pub static EPSILON: f32 = 1.19209290e-07_f32;

/// Smallest finite f32 value
pub static MIN_VALUE: f32 = -3.40282347e+38_f32;
/// Smallest positive, normalized f32 value
pub static MIN_POS_VALUE: f32 = 1.17549435e-38_f32;
/// Largest finite f32 value
pub static MAX_VALUE: f32 = 3.40282347e+38_f32;

pub static MIN_EXP: int = -125;
pub static MAX_EXP: int = 128;

pub static MIN_10_EXP: int = -37;
pub static MAX_10_EXP: int = 38;

pub static NAN: f32 = 0.0_f32/0.0_f32;
pub static INFINITY: f32 = 1.0_f32/0.0_f32;
pub static NEG_INFINITY: f32 = -1.0_f32/0.0_f32;

/// Various useful constants.
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    // FIXME(#5527): These constants should be deprecated once associated
    // constants are implemented in favour of referencing the respective members
    // of `Float`.

    /// Archimedes' constant
    pub static PI: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi * 2.0
    pub static PI_2: f32 = 6.28318530717958647692528676655900576_f32;

    /// pi/2.0
    pub static FRAC_PI_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/3.0
    pub static FRAC_PI_3: f32 = 1.04719755119659774615421446109316763_f32;

    /// pi/4.0
    pub static FRAC_PI_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// pi/6.0
    pub static FRAC_PI_6: f32 = 0.52359877559829887307710723054658381_f32;

    /// pi/8.0
    pub static FRAC_PI_8: f32 = 0.39269908169872415480783042290993786_f32;

    /// 1.0/pi
    pub static FRAC_1_PI: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2.0/pi
    pub static FRAC_2_PI: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2.0/sqrt(pi)
    pub static FRAC_2_SQRTPI: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2.0)
    pub static SQRT2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1.0/sqrt(2.0)
    pub static FRAC_1_SQRT2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number
    pub static E: f32 = 2.71828182845904523536028747135266250_f32;

    /// log2(e)
    pub static LOG2_E: f32 = 1.44269504088896340735992468100189214_f32;

    /// log10(e)
    pub static LOG10_E: f32 = 0.434294481903251827651128918916605082_f32;

    /// ln(2.0)
    pub static LN_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10.0)
    pub static LN_10: f32 = 2.30258509299404568401799145468436421_f32;
}

#[cfg(not(test))]
impl Ord for f32 {
    #[inline]
    fn lt(&self, other: &f32) -> bool { (*self) < (*other) }
    #[inline]
    fn le(&self, other: &f32) -> bool { (*self) <= (*other) }
    #[inline]
    fn ge(&self, other: &f32) -> bool { (*self) >= (*other) }
    #[inline]
    fn gt(&self, other: &f32) -> bool { (*self) > (*other) }
}
#[cfg(not(test))]
impl Eq for f32 {
    #[inline]
    fn eq(&self, other: &f32) -> bool { (*self) == (*other) }
}

impl Num for f32 {}

impl Default for f32 {
    #[inline]
    fn default() -> f32 { 0.0 }
}

impl Primitive for f32 {}

impl Zero for f32 {
    #[inline]
    fn zero() -> f32 { 0.0 }

    /// Returns true if the number is equal to either `0.0` or `-0.0`
    #[inline]
    fn is_zero(&self) -> bool { *self == 0.0 || *self == -0.0 }
}

impl One for f32 {
    #[inline]
    fn one() -> f32 { 1.0 }
}

#[cfg(not(test))]
impl Add<f32,f32> for f32 {
    #[inline]
    fn add(&self, other: &f32) -> f32 { *self + *other }
}

#[cfg(not(test))]
impl Sub<f32,f32> for f32 {
    #[inline]
    fn sub(&self, other: &f32) -> f32 { *self - *other }
}

#[cfg(not(test))]
impl Mul<f32,f32> for f32 {
    #[inline]
    fn mul(&self, other: &f32) -> f32 { *self * *other }
}

#[cfg(not(test))]
impl Div<f32,f32> for f32 {
    #[inline]
    fn div(&self, other: &f32) -> f32 { *self / *other }
}

#[cfg(not(test))]
impl Rem<f32,f32> for f32 {
    #[inline]
    fn rem(&self, other: &f32) -> f32 {
        extern { fn fmodf(a: f32, b: f32) -> f32; }
        unsafe { fmodf(*self, *other) }
    }
}

#[cfg(not(test))]
impl Neg<f32> for f32 {
    #[inline]
    fn neg(&self) -> f32 { -*self }
}

impl Signed for f32 {
    /// Computes the absolute value. Returns `NAN` if the number is `NAN`.
    #[inline]
    fn abs(&self) -> f32 {
        unsafe { intrinsics::fabsf32(*self) }
    }

    /// The positive difference of two numbers. Returns `0.0` if the number is
    /// less than or equal to `other`, otherwise the difference between`self`
    /// and `other` is returned.
    #[inline]
    fn abs_sub(&self, other: &f32) -> f32 {
        extern { fn fdimf(a: f32, b: f32) -> f32; }
        unsafe { fdimf(*self, *other) }
    }

    /// # Returns
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is NaN
    #[inline]
    fn signum(&self) -> f32 {
        if self != self { NAN } else {
            unsafe { intrinsics::copysignf32(1.0, *self) }
        }
    }

    /// Returns `true` if the number is positive, including `+0.0` and `INFINITY`
    #[inline]
    fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == INFINITY }

    /// Returns `true` if the number is negative, including `-0.0` and `NEG_INFINITY`
    #[inline]
    fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == NEG_INFINITY }
}

impl Bounded for f32 {
    // NOTE: this is the smallest non-infinite f32 value, *not* MIN_VALUE
    #[inline]
    fn min_value() -> f32 { -MAX_VALUE }

    #[inline]
    fn max_value() -> f32 { MAX_VALUE }
}

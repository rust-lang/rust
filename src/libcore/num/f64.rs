// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! Operations and constants for `f64`

use cmath;
use cmp;
use libc::{c_double, c_int};
use libc;
use num;

pub use cmath::c_double_targ_consts::*;

macro_rules! delegate(
    (
        fn $name:ident(
            $(
                $arg:ident : $arg_ty:ty
            ),*
        ) -> $rv:ty = $bound_name:path
    ) => (
        pub pure fn $name($( $arg : $arg_ty ),*) -> $rv {
            unsafe {
                $bound_name($( $arg ),*)
            }
        }
    )
)

delegate!(fn acos(n: c_double) -> c_double = cmath::c_double_utils::acos)
delegate!(fn asin(n: c_double) -> c_double = cmath::c_double_utils::asin)
delegate!(fn atan(n: c_double) -> c_double = cmath::c_double_utils::atan)
delegate!(fn atan2(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::atan2)
delegate!(fn cbrt(n: c_double) -> c_double = cmath::c_double_utils::cbrt)
delegate!(fn ceil(n: c_double) -> c_double = cmath::c_double_utils::ceil)
delegate!(fn copysign(x: c_double, y: c_double) -> c_double =
    cmath::c_double_utils::copysign)
delegate!(fn cos(n: c_double) -> c_double = cmath::c_double_utils::cos)
delegate!(fn cosh(n: c_double) -> c_double = cmath::c_double_utils::cosh)
delegate!(fn erf(n: c_double) -> c_double = cmath::c_double_utils::erf)
delegate!(fn erfc(n: c_double) -> c_double = cmath::c_double_utils::erfc)
delegate!(fn exp(n: c_double) -> c_double = cmath::c_double_utils::exp)
delegate!(fn expm1(n: c_double) -> c_double = cmath::c_double_utils::expm1)
delegate!(fn exp2(n: c_double) -> c_double = cmath::c_double_utils::exp2)
delegate!(fn abs(n: c_double) -> c_double = cmath::c_double_utils::abs)
delegate!(fn abs_sub(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::abs_sub)
delegate!(fn mul_add(a: c_double, b: c_double, c: c_double) -> c_double =
    cmath::c_double_utils::mul_add)
delegate!(fn fmax(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::fmax)
delegate!(fn fmin(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::fmin)
delegate!(fn nextafter(x: c_double, y: c_double) -> c_double =
    cmath::c_double_utils::nextafter)
delegate!(fn frexp(n: c_double, value: &mut c_int) -> c_double =
    cmath::c_double_utils::frexp)
delegate!(fn hypot(x: c_double, y: c_double) -> c_double =
    cmath::c_double_utils::hypot)
delegate!(fn ldexp(x: c_double, n: c_int) -> c_double =
    cmath::c_double_utils::ldexp)
delegate!(fn lgamma(n: c_double, sign: &mut c_int) -> c_double =
    cmath::c_double_utils::lgamma)
delegate!(fn ln(n: c_double) -> c_double = cmath::c_double_utils::ln)
delegate!(fn log_radix(n: c_double) -> c_double =
    cmath::c_double_utils::log_radix)
delegate!(fn ln1p(n: c_double) -> c_double = cmath::c_double_utils::ln1p)
delegate!(fn log10(n: c_double) -> c_double = cmath::c_double_utils::log10)
delegate!(fn log2(n: c_double) -> c_double = cmath::c_double_utils::log2)
delegate!(fn ilog_radix(n: c_double) -> c_int =
    cmath::c_double_utils::ilog_radix)
delegate!(fn modf(n: c_double, iptr: &mut c_double) -> c_double =
    cmath::c_double_utils::modf)
delegate!(fn pow(n: c_double, e: c_double) -> c_double =
    cmath::c_double_utils::pow)
delegate!(fn round(n: c_double) -> c_double = cmath::c_double_utils::round)
delegate!(fn ldexp_radix(n: c_double, i: c_int) -> c_double =
    cmath::c_double_utils::ldexp_radix)
delegate!(fn sin(n: c_double) -> c_double = cmath::c_double_utils::sin)
delegate!(fn sinh(n: c_double) -> c_double = cmath::c_double_utils::sinh)
delegate!(fn sqrt(n: c_double) -> c_double = cmath::c_double_utils::sqrt)
delegate!(fn tan(n: c_double) -> c_double = cmath::c_double_utils::tan)
delegate!(fn tanh(n: c_double) -> c_double = cmath::c_double_utils::tanh)
delegate!(fn tgamma(n: c_double) -> c_double = cmath::c_double_utils::tgamma)
delegate!(fn trunc(n: c_double) -> c_double = cmath::c_double_utils::trunc)
delegate!(fn j0(n: c_double) -> c_double = cmath::c_double_utils::j0)
delegate!(fn j1(n: c_double) -> c_double = cmath::c_double_utils::j1)
delegate!(fn jn(i: c_int, n: c_double) -> c_double =
    cmath::c_double_utils::jn)
delegate!(fn y0(n: c_double) -> c_double = cmath::c_double_utils::y0)
delegate!(fn y1(n: c_double) -> c_double = cmath::c_double_utils::y1)
delegate!(fn yn(i: c_int, n: c_double) -> c_double =
    cmath::c_double_utils::yn)

// FIXME (#1433): obtain these in a different way

// These are not defined inside consts:: for consistency with
// the integer types

pub const radix: uint = 2u;

pub const mantissa_digits: uint = 53u;
pub const digits: uint = 15u;

pub const epsilon: f64 = 2.2204460492503131e-16_f64;

pub const min_value: f64 = 2.2250738585072014e-308_f64;
pub const max_value: f64 = 1.7976931348623157e+308_f64;

pub const min_exp: int = -1021;
pub const max_exp: int = 1024;

pub const min_10_exp: int = -307;
pub const max_10_exp: int = 308;

pub const NaN: f64 = 0.0_f64/0.0_f64;

pub const infinity: f64 = 1.0_f64/0.0_f64;

pub const neg_infinity: f64 = -1.0_f64/0.0_f64;

#[inline(always)]
pub pure fn is_NaN(f: f64) -> bool { f != f }

#[inline(always)]
pub pure fn add(x: f64, y: f64) -> f64 { return x + y; }

#[inline(always)]
pub pure fn sub(x: f64, y: f64) -> f64 { return x - y; }

#[inline(always)]
pub pure fn mul(x: f64, y: f64) -> f64 { return x * y; }

#[inline(always)]
pub pure fn div(x: f64, y: f64) -> f64 { return x / y; }

#[inline(always)]
pub pure fn rem(x: f64, y: f64) -> f64 { return x % y; }

#[inline(always)]
pub pure fn lt(x: f64, y: f64) -> bool { return x < y; }

#[inline(always)]
pub pure fn le(x: f64, y: f64) -> bool { return x <= y; }

#[inline(always)]
pub pure fn eq(x: f64, y: f64) -> bool { return x == y; }

#[inline(always)]
pub pure fn ne(x: f64, y: f64) -> bool { return x != y; }

#[inline(always)]
pub pure fn ge(x: f64, y: f64) -> bool { return x >= y; }

#[inline(always)]
pub pure fn gt(x: f64, y: f64) -> bool { return x > y; }

/// Returns true if `x` is a positive number, including +0.0f640 and +Infinity
#[inline(always)]
pub pure fn is_positive(x: f64) -> bool
    { return x > 0.0f64 || (1.0f64/x) == infinity; }

/// Returns true if `x` is a negative number, including -0.0f640 and -Infinity
#[inline(always)]
pub pure fn is_negative(x: f64) -> bool
    { return x < 0.0f64 || (1.0f64/x) == neg_infinity; }

/**
 * Returns true if `x` is a negative number, including -0.0f640 and -Infinity
 *
 * This is the same as `f64::is_negative`.
 */
#[inline(always)]
pub pure fn is_nonpositive(x: f64) -> bool {
  return x < 0.0f64 || (1.0f64/x) == neg_infinity;
}

/**
 * Returns true if `x` is a positive number, including +0.0f640 and +Infinity
 *
 * This is the same as `f64::positive`.
 */
#[inline(always)]
pub pure fn is_nonnegative(x: f64) -> bool {
  return x > 0.0f64 || (1.0f64/x) == infinity;
}

/// Returns true if `x` is a zero number (positive or negative zero)
#[inline(always)]
pub pure fn is_zero(x: f64) -> bool {
    return x == 0.0f64 || x == -0.0f64;
}

/// Returns true if `x`is an infinite number
#[inline(always)]
pub pure fn is_infinite(x: f64) -> bool {
    return x == infinity || x == neg_infinity;
}

/// Returns true if `x` is a finite number
#[inline(always)]
pub pure fn is_finite(x: f64) -> bool {
    return !(is_NaN(x) || is_infinite(x));
}

/// Returns `x` rounded down
#[inline(always)]
pub pure fn floor(x: f64) -> f64 { unsafe { floorf64(x) } }

// FIXME (#1999): add is_normal, is_subnormal, and fpclassify

/* Module: consts */
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // constants from cmath.
    /// Archimedes' constant
    pub const pi: f64 = 3.14159265358979323846264338327950288_f64;

    /// pi/2.0
    pub const frac_pi_2: f64 = 1.57079632679489661923132169163975144_f64;

    /// pi/4.0
    pub const frac_pi_4: f64 = 0.785398163397448309615660845819875721_f64;

    /// 1.0/pi
    pub const frac_1_pi: f64 = 0.318309886183790671537767526745028724_f64;

    /// 2.0/pi
    pub const frac_2_pi: f64 = 0.636619772367581343075535053490057448_f64;

    /// 2.0/sqrt(pi)
    pub const frac_2_sqrtpi: f64 = 1.12837916709551257389615890312154517_f64;

    /// sqrt(2.0)
    pub const sqrt2: f64 = 1.41421356237309504880168872420969808_f64;

    /// 1.0/sqrt(2.0)
    pub const frac_1_sqrt2: f64 = 0.707106781186547524400844362104849039_f64;

    /// Euler's number
    pub const e: f64 = 2.71828182845904523536028747135266250_f64;

    /// log2(e)
    pub const log2_e: f64 = 1.44269504088896340735992468100189214_f64;

    /// log10(e)
    pub const log10_e: f64 = 0.434294481903251827651128918916605082_f64;

    /// ln(2.0)
    pub const ln_2: f64 = 0.693147180559945309417232121458176568_f64;

    /// ln(10.0)
    pub const ln_10: f64 = 2.30258509299404568401799145468436421_f64;
}

#[inline(always)]
pub pure fn signbit(x: f64) -> int {
    if is_negative(x) { return 1; } else { return 0; }
}

#[inline(always)]
pub pure fn logarithm(n: f64, b: f64) -> f64 {
    return log2(n) / log2(b);
}

#[cfg(notest)]
impl f64 : cmp::Eq {
    #[inline(always)]
    pure fn eq(&self, other: &f64) -> bool { (*self) == (*other) }
    #[inline(always)]
    pure fn ne(&self, other: &f64) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl f64 : cmp::Ord {
    #[inline(always)]
    pure fn lt(&self, other: &f64) -> bool { (*self) < (*other) }
    #[inline(always)]
    pure fn le(&self, other: &f64) -> bool { (*self) <= (*other) }
    #[inline(always)]
    pure fn ge(&self, other: &f64) -> bool { (*self) >= (*other) }
    #[inline(always)]
    pure fn gt(&self, other: &f64) -> bool { (*self) > (*other) }
}

impl f64: num::Num {
    #[inline(always)]
    pure fn add(&self, other: &f64)    -> f64 { return *self + *other; }
    #[inline(always)]
    pure fn sub(&self, other: &f64)    -> f64 { return *self - *other; }
    #[inline(always)]
    pure fn mul(&self, other: &f64)    -> f64 { return *self * *other; }
    #[inline(always)]
    pure fn div(&self, other: &f64)    -> f64 { return *self / *other; }
    #[inline(always)]
    pure fn modulo(&self, other: &f64) -> f64 { return *self % *other; }
    #[inline(always)]
    pure fn neg(&self)                -> f64 { return -*self;        }

    #[inline(always)]
    pure fn to_int(&self)         -> int { return *self as int; }
    #[inline(always)]
    static pure fn from_int(n: int) -> f64 { return n as f64;    }
}

impl f64: num::Zero {
    #[inline(always)]
    static pure fn zero() -> f64 { 0.0 }
}

impl f64: num::One {
    #[inline(always)]
    static pure fn one() -> f64 { 1.0 }
}

#[abi="rust-intrinsic"]
pub extern {
    fn floorf64(val: f64) -> f64;
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

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

//! Operations and constants for `f32`

use cmath;
use cmp;
use libc::{c_float, c_int};
use num;

pub use cmath::c_float_targ_consts::*;

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

delegate!(fn acos(n: c_float) -> c_float = cmath::c_float_utils::acos)
delegate!(fn asin(n: c_float) -> c_float = cmath::c_float_utils::asin)
delegate!(fn atan(n: c_float) -> c_float = cmath::c_float_utils::atan)
delegate!(fn atan2(a: c_float, b: c_float) -> c_float =
    cmath::c_float_utils::atan2)
delegate!(fn cbrt(n: c_float) -> c_float = cmath::c_float_utils::cbrt)
delegate!(fn ceil(n: c_float) -> c_float = cmath::c_float_utils::ceil)
delegate!(fn copysign(x: c_float, y: c_float) -> c_float =
    cmath::c_float_utils::copysign)
delegate!(fn cos(n: c_float) -> c_float = cmath::c_float_utils::cos)
delegate!(fn cosh(n: c_float) -> c_float = cmath::c_float_utils::cosh)
delegate!(fn erf(n: c_float) -> c_float = cmath::c_float_utils::erf)
delegate!(fn erfc(n: c_float) -> c_float = cmath::c_float_utils::erfc)
delegate!(fn exp(n: c_float) -> c_float = cmath::c_float_utils::exp)
delegate!(fn expm1(n: c_float) -> c_float = cmath::c_float_utils::expm1)
delegate!(fn exp2(n: c_float) -> c_float = cmath::c_float_utils::exp2)
delegate!(fn abs(n: c_float) -> c_float = cmath::c_float_utils::abs)
delegate!(fn abs_sub(a: c_float, b: c_float) -> c_float =
    cmath::c_float_utils::abs_sub)
delegate!(fn mul_add(a: c_float, b: c_float, c: c_float) -> c_float =
    cmath::c_float_utils::mul_add)
delegate!(fn fmax(a: c_float, b: c_float) -> c_float =
    cmath::c_float_utils::fmax)
delegate!(fn fmin(a: c_float, b: c_float) -> c_float =
    cmath::c_float_utils::fmin)
delegate!(fn nextafter(x: c_float, y: c_float) -> c_float =
    cmath::c_float_utils::nextafter)
delegate!(fn frexp(n: c_float, value: &mut c_int) -> c_float =
    cmath::c_float_utils::frexp)
delegate!(fn hypot(x: c_float, y: c_float) -> c_float =
    cmath::c_float_utils::hypot)
delegate!(fn ldexp(x: c_float, n: c_int) -> c_float =
    cmath::c_float_utils::ldexp)
delegate!(fn lgamma(n: c_float, sign: &mut c_int) -> c_float =
    cmath::c_float_utils::lgamma)
delegate!(fn ln(n: c_float) -> c_float = cmath::c_float_utils::ln)
delegate!(fn log_radix(n: c_float) -> c_float =
    cmath::c_float_utils::log_radix)
delegate!(fn ln1p(n: c_float) -> c_float = cmath::c_float_utils::ln1p)
delegate!(fn log10(n: c_float) -> c_float = cmath::c_float_utils::log10)
delegate!(fn log2(n: c_float) -> c_float = cmath::c_float_utils::log2)
delegate!(fn ilog_radix(n: c_float) -> c_int =
    cmath::c_float_utils::ilog_radix)
delegate!(fn modf(n: c_float, iptr: &mut c_float) -> c_float =
    cmath::c_float_utils::modf)
delegate!(fn pow(n: c_float, e: c_float) -> c_float =
    cmath::c_float_utils::pow)
delegate!(fn round(n: c_float) -> c_float = cmath::c_float_utils::round)
delegate!(fn ldexp_radix(n: c_float, i: c_int) -> c_float =
    cmath::c_float_utils::ldexp_radix)
delegate!(fn sin(n: c_float) -> c_float = cmath::c_float_utils::sin)
delegate!(fn sinh(n: c_float) -> c_float = cmath::c_float_utils::sinh)
delegate!(fn sqrt(n: c_float) -> c_float = cmath::c_float_utils::sqrt)
delegate!(fn tan(n: c_float) -> c_float = cmath::c_float_utils::tan)
delegate!(fn tanh(n: c_float) -> c_float = cmath::c_float_utils::tanh)
delegate!(fn tgamma(n: c_float) -> c_float = cmath::c_float_utils::tgamma)
delegate!(fn trunc(n: c_float) -> c_float = cmath::c_float_utils::trunc)

// These are not defined inside consts:: for consistency with
// the integer types

pub const NaN: f32 = 0.0_f32/0.0_f32;

pub const infinity: f32 = 1.0_f32/0.0_f32;

pub const neg_infinity: f32 = -1.0_f32/0.0_f32;

#[inline(always)]
pub pure fn is_NaN(f: f32) -> bool { f != f }

#[inline(always)]
pub pure fn add(x: f32, y: f32) -> f32 { return x + y; }

#[inline(always)]
pub pure fn sub(x: f32, y: f32) -> f32 { return x - y; }

#[inline(always)]
pub pure fn mul(x: f32, y: f32) -> f32 { return x * y; }

#[inline(always)]
pub pure fn div(x: f32, y: f32) -> f32 { return x / y; }

#[inline(always)]
pub pure fn rem(x: f32, y: f32) -> f32 { return x % y; }

#[inline(always)]
pub pure fn lt(x: f32, y: f32) -> bool { return x < y; }

#[inline(always)]
pub pure fn le(x: f32, y: f32) -> bool { return x <= y; }

#[inline(always)]
pub pure fn eq(x: f32, y: f32) -> bool { return x == y; }

#[inline(always)]
pub pure fn ne(x: f32, y: f32) -> bool { return x != y; }

#[inline(always)]
pub pure fn ge(x: f32, y: f32) -> bool { return x >= y; }

#[inline(always)]
pub pure fn gt(x: f32, y: f32) -> bool { return x > y; }

/// Returns `x` rounded down
#[inline(always)]
pub pure fn floor(x: f32) -> f32 { unsafe { floorf32(x) } }

// FIXME (#1999): replace the predicates below with llvm intrinsics or
// calls to the libmath macros in the rust runtime for performance.

/// Returns true if `x` is a positive number, including +0.0f320 and +Infinity
#[inline(always)]
pub pure fn is_positive(x: f32) -> bool
    { return x > 0.0f32 || (1.0f32/x) == infinity; }

/// Returns true if `x` is a negative number, including -0.0f320 and -Infinity
#[inline(always)]
pub pure fn is_negative(x: f32) -> bool
    { return x < 0.0f32 || (1.0f32/x) == neg_infinity; }

/**
 * Returns true if `x` is a negative number, including -0.0f320 and -Infinity
 *
 * This is the same as `f32::is_negative`.
 */
#[inline(always)]
pub pure fn is_nonpositive(x: f32) -> bool {
  return x < 0.0f32 || (1.0f32/x) == neg_infinity;
}

/**
 * Returns true if `x` is a positive number, including +0.0f320 and +Infinity
 *
 * This is the same as `f32::is_positive`.)
 */
#[inline(always)]
pub pure fn is_nonnegative(x: f32) -> bool {
  return x > 0.0f32 || (1.0f32/x) == infinity;
}

/// Returns true if `x` is a zero number (positive or negative zero)
#[inline(always)]
pub pure fn is_zero(x: f32) -> bool {
    return x == 0.0f32 || x == -0.0f32;
}

/// Returns true if `x`is an infinite number
#[inline(always)]
pub pure fn is_infinite(x: f32) -> bool {
    return x == infinity || x == neg_infinity;
}

/// Returns true if `x`is a finite number
#[inline(always)]
pub pure fn is_finite(x: f32) -> bool {
    return !(is_NaN(x) || is_infinite(x));
}

// FIXME (#1999): add is_normal, is_subnormal, and fpclassify.

/* Module: consts */
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // constants from cmath.
    /// Archimedes' constant
    pub const pi: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi/2.0
    pub const frac_pi_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/4.0
    pub const frac_pi_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// 1.0/pi
    pub const frac_1_pi: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2.0/pi
    pub const frac_2_pi: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2.0/sqrt(pi)
    pub const frac_2_sqrtpi: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2.0)
    pub const sqrt2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1.0/sqrt(2.0)
    pub const frac_1_sqrt2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number
    pub const e: f32 = 2.71828182845904523536028747135266250_f32;

    /// log2(e)
    pub const log2_e: f32 = 1.44269504088896340735992468100189214_f32;

    /// log10(e)
    pub const log10_e: f32 = 0.434294481903251827651128918916605082_f32;

    /// ln(2.0)
    pub const ln_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10.0)
    pub const ln_10: f32 = 2.30258509299404568401799145468436421_f32;
}

#[inline(always)]
pub pure fn signbit(x: f32) -> int {
    if is_negative(x) { return 1; } else { return 0; }
}

#[inline(always)]
pub pure fn logarithm(n: f32, b: f32) -> f32 {
    return log2(n) / log2(b);
}

#[cfg(notest)]
impl f32 : cmp::Eq {
    #[inline(always)]
    pure fn eq(&self, other: &f32) -> bool { (*self) == (*other) }
    #[inline(always)]
    pure fn ne(&self, other: &f32) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl f32 : cmp::Ord {
    #[inline(always)]
    pure fn lt(&self, other: &f32) -> bool { (*self) < (*other) }
    #[inline(always)]
    pure fn le(&self, other: &f32) -> bool { (*self) <= (*other) }
    #[inline(always)]
    pure fn ge(&self, other: &f32) -> bool { (*self) >= (*other) }
    #[inline(always)]
    pure fn gt(&self, other: &f32) -> bool { (*self) > (*other) }
}

impl f32: num::Num {
    #[inline(always)]
    pure fn add(&self, other: &f32) -> f32 { return *self + *other; }
    #[inline(always)]
    pure fn sub(&self, other: &f32) -> f32 { return *self - *other; }
    #[inline(always)]
    pure fn mul(&self, other: &f32) -> f32 { return *self * *other; }
    #[inline(always)]
    pure fn div(&self, other: &f32) -> f32 { return *self / *other; }
    #[inline(always)]
    pure fn modulo(&self, other: &f32) -> f32 { return *self % *other; }
    #[inline(always)]
    pure fn neg(&self)                -> f32 { return -*self;        }

    #[inline(always)]
    pure fn to_int(&self)         -> int { return *self as int; }
    #[inline(always)]
    static pure fn from_int(n: int) -> f32 { return n as f32;    }
}

impl f32: num::Zero {
    #[inline(always)]
    static pure fn zero() -> f32 { 0.0 }
}

impl f32: num::One {
    #[inline(always)]
    static pure fn one() -> f32 { 1.0 }
}

#[abi="rust-intrinsic"]
pub extern {
    fn floorf32(val: f32) -> f32;
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

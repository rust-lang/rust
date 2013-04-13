// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `float`

// Even though this module exports everything defined in it,
// because it contains re-exports, we also have to explicitly
// export locally defined things. That's a bit annoying.


// export when m_float == c_double


// PORT this must match in width according to architecture

use f64;
use num::strconv;
use num;
use option::Option;
use to_str;
use from_str;

#[cfg(notest)] use cmp::{Eq, Ord};
#[cfg(notest)] use ops;
#[cfg(test)] use option::{Some, None};

pub use f64::{add, sub, mul, div, rem, lt, le, eq, ne, ge, gt};
pub use f64::logarithm;
pub use f64::{acos, asin, atan2, cbrt, ceil, copysign, cosh, floor};
pub use f64::{erf, erfc, exp, expm1, exp2, abs_sub};
pub use f64::{mul_add, fmax, fmin, nextafter, frexp, hypot, ldexp};
pub use f64::{lgamma, ln, log_radix, ln1p, log10, log2, ilog_radix};
pub use f64::{modf, pow, round, sinh, tanh, tgamma, trunc};
pub use f64::signbit;
pub use f64::{j0, j1, jn, y0, y1, yn};

pub static NaN: float = 0.0/0.0;

pub static infinity: float = 1.0/0.0;

pub static neg_infinity: float = -1.0/0.0;

/* Module: consts */
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // staticants from cmath.
    /// Archimedes' staticant
    pub static pi: float = 3.14159265358979323846264338327950288;

    /// pi/2.0
    pub static frac_pi_2: float = 1.57079632679489661923132169163975144;

    /// pi/4.0
    pub static frac_pi_4: float = 0.785398163397448309615660845819875721;

    /// 1.0/pi
    pub static frac_1_pi: float = 0.318309886183790671537767526745028724;

    /// 2.0/pi
    pub static frac_2_pi: float = 0.636619772367581343075535053490057448;

    /// 2.0/sqrt(pi)
    pub static frac_2_sqrtpi: float = 1.12837916709551257389615890312154517;

    /// sqrt(2.0)
    pub static sqrt2: float = 1.41421356237309504880168872420969808;

    /// 1.0/sqrt(2.0)
    pub static frac_1_sqrt2: float = 0.707106781186547524400844362104849039;

    /// Euler's number
    pub static e: float = 2.71828182845904523536028747135266250;

    /// log2(e)
    pub static log2_e: float = 1.44269504088896340735992468100189214;

    /// log10(e)
    pub static log10_e: float = 0.434294481903251827651128918916605082;

    /// ln(2.0)
    pub static ln_2: float = 0.693147180559945309417232121458176568;

    /// ln(10.0)
    pub static ln_10: float = 2.30258509299404568401799145468436421;
}

/*
 * Section: String Conversions
 */

/**
 * Converts a float to a string
 *
 * # Arguments
 *
 * * num - The float value
 */
#[inline(always)]
pub fn to_str(num: float) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigAll);
    r
}

/**
 * Converts a float to a string in hexadecimal format
 *
 * # Arguments
 *
 * * num - The float value
 */
#[inline(always)]
pub fn to_str_hex(num: float) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 16u, true, strconv::SignNeg, strconv::DigAll);
    r
}

/**
 * Converts a float to a string in a given radix
 *
 * # Arguments
 *
 * * num - The float value
 * * radix - The base to use
 *
 * # Failure
 *
 * Fails if called on a special value like `inf`, `-inf` or `NaN` due to
 * possible misinterpretation of the result at higher bases. If those values
 * are expected, use `to_str_radix_special()` instead.
 */
#[inline(always)]
pub fn to_str_radix(num: float, radix: uint) -> ~str {
    let (r, special) = strconv::to_str_common(
        &num, radix, true, strconv::SignNeg, strconv::DigAll);
    if special { fail!(~"number has a special value, \
                      try to_str_radix_special() if those are expected") }
    r
}

/**
 * Converts a float to a string in a given radix, and a flag indicating
 * whether it's a special value
 *
 * # Arguments
 *
 * * num - The float value
 * * radix - The base to use
 */
#[inline(always)]
pub fn to_str_radix_special(num: float, radix: uint) -> (~str, bool) {
    strconv::to_str_common(&num, radix, true,
                           strconv::SignNeg, strconv::DigAll)
}

/**
 * Converts a float to a string with exactly the number of
 * provided significant digits
 *
 * # Arguments
 *
 * * num - The float value
 * * digits - The number of significant digits
 */
#[inline(always)]
pub fn to_str_exact(num: float, digits: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigExact(digits));
    r
}

#[test]
pub fn test_to_str_exact_do_decimal() {
    let s = to_str_exact(5.0, 4u);
    assert!(s == ~"5.0000");
}

/**
 * Converts a float to a string with a maximum number of
 * significant digits
 *
 * # Arguments
 *
 * * num - The float value
 * * digits - The number of significant digits
 */
#[inline(always)]
pub fn to_str_digits(num: float, digits: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigMax(digits));
    r
}

impl to_str::ToStr for float {
    #[inline(always)]
    fn to_str(&self) -> ~str { to_str_digits(*self, 8) }
}

impl num::ToStrRadix for float {
    #[inline(always)]
    fn to_str_radix(&self, radix: uint) -> ~str {
        to_str_radix(*self, radix)
    }
}

/**
 * Convert a string in base 10 to a float.
 * Accepts a optional decimal exponent.
 *
 * This function accepts strings such as
 *
 * * '3.14'
 * * '+3.14', equivalent to '3.14'
 * * '-3.14'
 * * '2.5E10', or equivalently, '2.5e10'
 * * '2.5E-10'
 * * '.' (understood as 0)
 * * '5.'
 * * '.5', or, equivalently,  '0.5'
 * * '+inf', 'inf', '-inf', 'NaN'
 *
 * Leading and trailing whitespace represent an error.
 *
 * # Arguments
 *
 * * num - A string
 *
 * # Return value
 *
 * `none` if the string did not represent a valid number.  Otherwise,
 * `Some(n)` where `n` is the floating-point number represented by `num`.
 */
#[inline(always)]
pub fn from_str(num: &str) -> Option<float> {
    strconv::from_str_common(num, 10u, true, true, true,
                             strconv::ExpDec, false, false)
}

/**
 * Convert a string in base 16 to a float.
 * Accepts a optional binary exponent.
 *
 * This function accepts strings such as
 *
 * * 'a4.fe'
 * * '+a4.fe', equivalent to 'a4.fe'
 * * '-a4.fe'
 * * '2b.aP128', or equivalently, '2b.ap128'
 * * '2b.aP-128'
 * * '.' (understood as 0)
 * * 'c.'
 * * '.c', or, equivalently,  '0.c'
 * * '+inf', 'inf', '-inf', 'NaN'
 *
 * Leading and trailing whitespace represent an error.
 *
 * # Arguments
 *
 * * num - A string
 *
 * # Return value
 *
 * `none` if the string did not represent a valid number.  Otherwise,
 * `Some(n)` where `n` is the floating-point number represented by `[num]`.
 */
#[inline(always)]
pub fn from_str_hex(num: &str) -> Option<float> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

/**
 * Convert a string in an given base to a float.
 *
 * Due to possible conflicts, this function does **not** accept
 * the special values `inf`, `-inf`, `+inf` and `NaN`, **nor**
 * does it recognize exponents of any kind.
 *
 * Leading and trailing whitespace represent an error.
 *
 * # Arguments
 *
 * * num - A string
 * * radix - The base to use. Must lie in the range [2 .. 36]
 *
 * # Return value
 *
 * `none` if the string did not represent a valid number. Otherwise,
 * `Some(n)` where `n` is the floating-point number represented by `num`.
 */
#[inline(always)]
pub fn from_str_radix(num: &str, radix: uint) -> Option<float> {
    strconv::from_str_common(num, radix, true, true, false,
                             strconv::ExpNone, false, false)
}

impl from_str::FromStr for float {
    #[inline(always)]
    fn from_str(val: &str) -> Option<float> { from_str(val) }
}

impl num::FromStrRadix for float {
    #[inline(always)]
    fn from_str_radix(val: &str, radix: uint) -> Option<float> {
        from_str_radix(val, radix)
    }
}

/**
 * Section: Arithmetics
 */

/**
 * Compute the exponentiation of an integer by another integer as a float
 *
 * # Arguments
 *
 * * x - The base
 * * pow - The exponent
 *
 * # Return value
 *
 * `NaN` if both `x` and `pow` are `0u`, otherwise `x^pow`
 */
pub fn pow_with_uint(base: uint, pow: uint) -> float {
    if base == 0u {
        if pow == 0u {
            return NaN as float;
        }
        return 0.;
    }
    let mut my_pow     = pow;
    let mut total      = 1f;
    let mut multiplier = base as float;
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total = total * multiplier;
        }
        my_pow     /= 2u;
        multiplier *= multiplier;
    }
    return total;
}

#[inline(always)]
pub fn is_positive(x: float) -> bool { f64::is_positive(x as f64) }
#[inline(always)]
pub fn is_negative(x: float) -> bool { f64::is_negative(x as f64) }
#[inline(always)]
pub fn is_nonpositive(x: float) -> bool { f64::is_nonpositive(x as f64) }
#[inline(always)]
pub fn is_nonnegative(x: float) -> bool { f64::is_nonnegative(x as f64) }
#[inline(always)]
pub fn is_zero(x: float) -> bool { f64::is_zero(x as f64) }
#[inline(always)]
pub fn is_infinite(x: float) -> bool { f64::is_infinite(x as f64) }
#[inline(always)]
pub fn is_finite(x: float) -> bool { f64::is_finite(x as f64) }
#[inline(always)]
pub fn is_NaN(x: float) -> bool { f64::is_NaN(x as f64) }

#[inline(always)]
pub fn abs(x: float) -> float {
    unsafe { f64::abs(x as f64) as float }
}
#[inline(always)]
pub fn sqrt(x: float) -> float {
    unsafe { f64::sqrt(x as f64) as float }
}
#[inline(always)]
pub fn atan(x: float) -> float {
    unsafe { f64::atan(x as f64) as float }
}
#[inline(always)]
pub fn sin(x: float) -> float {
    unsafe { f64::sin(x as f64) as float }
}
#[inline(always)]
pub fn cos(x: float) -> float {
    unsafe { f64::cos(x as f64) as float }
}
#[inline(always)]
pub fn tan(x: float) -> float {
    unsafe { f64::tan(x as f64) as float }
}

#[cfg(notest)]
impl Eq for float {
    fn eq(&self, other: &float) -> bool { (*self) == (*other) }
    fn ne(&self, other: &float) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl Ord for float {
    fn lt(&self, other: &float) -> bool { (*self) < (*other) }
    fn le(&self, other: &float) -> bool { (*self) <= (*other) }
    fn ge(&self, other: &float) -> bool { (*self) >= (*other) }
    fn gt(&self, other: &float) -> bool { (*self) > (*other) }
}

impl num::Zero for float {
    #[inline(always)]
    fn zero() -> float { 0.0 }
}

impl num::One for float {
    #[inline(always)]
    fn one() -> float { 1.0 }
}

impl num::Round for float {
    #[inline(always)]
    fn round(&self, mode: num::RoundMode) -> float {
        match mode {
            num::RoundDown
                => f64::floor(*self as f64) as float,
            num::RoundUp
                => f64::ceil(*self as f64) as float,
            num::RoundToZero   if is_negative(*self)
                => f64::ceil(*self as f64) as float,
            num::RoundToZero
                => f64::floor(*self as f64) as float,
            num::RoundFromZero if is_negative(*self)
                => f64::floor(*self as f64) as float,
            num::RoundFromZero
                => f64::ceil(*self as f64) as float
        }
    }

    #[inline(always)]
    fn floor(&self) -> float { f64::floor(*self as f64) as float}
    #[inline(always)]
    fn ceil(&self) -> float { f64::ceil(*self as f64) as float}
    #[inline(always)]
    fn fract(&self) -> float {
        if is_negative(*self) {
            (*self) - (f64::ceil(*self as f64) as float)
        } else {
            (*self) - (f64::floor(*self as f64) as float)
        }
    }
}

#[cfg(notest)]
impl ops::Add<float,float> for float {
    fn add(&self, other: &float) -> float { *self + *other }
}
#[cfg(notest)]
impl ops::Sub<float,float> for float {
    fn sub(&self, other: &float) -> float { *self - *other }
}
#[cfg(notest)]
impl ops::Mul<float,float> for float {
    fn mul(&self, other: &float) -> float { *self * *other }
}
#[cfg(notest)]
impl ops::Div<float,float> for float {
    fn div(&self, other: &float) -> float { *self / *other }
}
#[cfg(notest)]
impl ops::Modulo<float,float> for float {
    fn modulo(&self, other: &float) -> float { *self % *other }
}
#[cfg(notest)]
impl ops::Neg<float> for float {
    fn neg(&self) -> float { -*self }
}

#[test]
pub fn test_from_str() {
   assert!(from_str(~"3") == Some(3.));
   assert!(from_str(~"3.14") == Some(3.14));
   assert!(from_str(~"+3.14") == Some(3.14));
   assert!(from_str(~"-3.14") == Some(-3.14));
   assert!(from_str(~"2.5E10") == Some(25000000000.));
   assert!(from_str(~"2.5e10") == Some(25000000000.));
   assert!(from_str(~"25000000000.E-10") == Some(2.5));
   assert!(from_str(~".") == Some(0.));
   assert!(from_str(~".e1") == Some(0.));
   assert!(from_str(~".e-1") == Some(0.));
   assert!(from_str(~"5.") == Some(5.));
   assert!(from_str(~".5") == Some(0.5));
   assert!(from_str(~"0.5") == Some(0.5));
   assert!(from_str(~"-.5") == Some(-0.5));
   assert!(from_str(~"-5") == Some(-5.));
   assert!(from_str(~"inf") == Some(infinity));
   assert!(from_str(~"+inf") == Some(infinity));
   assert!(from_str(~"-inf") == Some(neg_infinity));
   // note: NaN != NaN, hence this slightly complex test
   match from_str(~"NaN") {
       Some(f) => assert!(is_NaN(f)),
       None => fail!()
   }
   // note: -0 == 0, hence these slightly more complex tests
   match from_str(~"-0") {
       Some(v) if is_zero(v) => assert!(is_negative(v)),
       _ => fail!()
   }
   match from_str(~"0") {
       Some(v) if is_zero(v) => assert!(is_positive(v)),
       _ => fail!()
   }

   assert!(from_str(~"").is_none());
   assert!(from_str(~"x").is_none());
   assert!(from_str(~" ").is_none());
   assert!(from_str(~"   ").is_none());
   assert!(from_str(~"e").is_none());
   assert!(from_str(~"E").is_none());
   assert!(from_str(~"E1").is_none());
   assert!(from_str(~"1e1e1").is_none());
   assert!(from_str(~"1e1.1").is_none());
   assert!(from_str(~"1e1-1").is_none());
}

#[test]
pub fn test_from_str_hex() {
   assert!(from_str_hex(~"a4") == Some(164.));
   assert!(from_str_hex(~"a4.fe") == Some(164.9921875));
   assert!(from_str_hex(~"-a4.fe") == Some(-164.9921875));
   assert!(from_str_hex(~"+a4.fe") == Some(164.9921875));
   assert!(from_str_hex(~"ff0P4") == Some(0xff00 as float));
   assert!(from_str_hex(~"ff0p4") == Some(0xff00 as float));
   assert!(from_str_hex(~"ff0p-4") == Some(0xff as float));
   assert!(from_str_hex(~".") == Some(0.));
   assert!(from_str_hex(~".p1") == Some(0.));
   assert!(from_str_hex(~".p-1") == Some(0.));
   assert!(from_str_hex(~"f.") == Some(15.));
   assert!(from_str_hex(~".f") == Some(0.9375));
   assert!(from_str_hex(~"0.f") == Some(0.9375));
   assert!(from_str_hex(~"-.f") == Some(-0.9375));
   assert!(from_str_hex(~"-f") == Some(-15.));
   assert!(from_str_hex(~"inf") == Some(infinity));
   assert!(from_str_hex(~"+inf") == Some(infinity));
   assert!(from_str_hex(~"-inf") == Some(neg_infinity));
   // note: NaN != NaN, hence this slightly complex test
   match from_str_hex(~"NaN") {
       Some(f) => assert!(is_NaN(f)),
       None => fail!()
   }
   // note: -0 == 0, hence these slightly more complex tests
   match from_str_hex(~"-0") {
       Some(v) if is_zero(v) => assert!(is_negative(v)),
       _ => fail!()
   }
   match from_str_hex(~"0") {
       Some(v) if is_zero(v) => assert!(is_positive(v)),
       _ => fail!()
   }
   assert!(from_str_hex(~"e") == Some(14.));
   assert!(from_str_hex(~"E") == Some(14.));
   assert!(from_str_hex(~"E1") == Some(225.));
   assert!(from_str_hex(~"1e1e1") == Some(123361.));
   assert!(from_str_hex(~"1e1.1") == Some(481.0625));

   assert!(from_str_hex(~"").is_none());
   assert!(from_str_hex(~"x").is_none());
   assert!(from_str_hex(~" ").is_none());
   assert!(from_str_hex(~"   ").is_none());
   assert!(from_str_hex(~"p").is_none());
   assert!(from_str_hex(~"P").is_none());
   assert!(from_str_hex(~"P1").is_none());
   assert!(from_str_hex(~"1p1p1").is_none());
   assert!(from_str_hex(~"1p1.1").is_none());
   assert!(from_str_hex(~"1p1-1").is_none());
}

#[test]
pub fn test_to_str_hex() {
   assert!(to_str_hex(164.) == ~"a4");
   assert!(to_str_hex(164.9921875) == ~"a4.fe");
   assert!(to_str_hex(-164.9921875) == ~"-a4.fe");
   assert!(to_str_hex(0xff00 as float) == ~"ff00");
   assert!(to_str_hex(-(0xff00 as float)) == ~"-ff00");
   assert!(to_str_hex(0.) == ~"0");
   assert!(to_str_hex(15.) == ~"f");
   assert!(to_str_hex(-15.) == ~"-f");
   assert!(to_str_hex(0.9375) == ~"0.f");
   assert!(to_str_hex(-0.9375) == ~"-0.f");
   assert!(to_str_hex(infinity) == ~"inf");
   assert!(to_str_hex(neg_infinity) == ~"-inf");
   assert!(to_str_hex(NaN) == ~"NaN");
   assert!(to_str_hex(0.) == ~"0");
   assert!(to_str_hex(-0.) == ~"-0");
}

#[test]
pub fn test_to_str_radix() {
   assert!(to_str_radix(36., 36u) == ~"10");
   assert!(to_str_radix(8.125, 2u) == ~"1000.001");
}

#[test]
pub fn test_from_str_radix() {
   assert!(from_str_radix(~"10", 36u) == Some(36.));
   assert!(from_str_radix(~"1000.001", 2u) == Some(8.125));
}

#[test]
pub fn test_positive() {
  assert!((is_positive(infinity)));
  assert!((is_positive(1.)));
  assert!((is_positive(0.)));
  assert!((!is_positive(-1.)));
  assert!((!is_positive(neg_infinity)));
  assert!((!is_positive(1./neg_infinity)));
  assert!((!is_positive(NaN)));
}

#[test]
pub fn test_negative() {
  assert!((!is_negative(infinity)));
  assert!((!is_negative(1.)));
  assert!((!is_negative(0.)));
  assert!((is_negative(-1.)));
  assert!((is_negative(neg_infinity)));
  assert!((is_negative(1./neg_infinity)));
  assert!((!is_negative(NaN)));
}

#[test]
pub fn test_nonpositive() {
  assert!((!is_nonpositive(infinity)));
  assert!((!is_nonpositive(1.)));
  assert!((!is_nonpositive(0.)));
  assert!((is_nonpositive(-1.)));
  assert!((is_nonpositive(neg_infinity)));
  assert!((is_nonpositive(1./neg_infinity)));
  assert!((!is_nonpositive(NaN)));
}

#[test]
pub fn test_nonnegative() {
  assert!((is_nonnegative(infinity)));
  assert!((is_nonnegative(1.)));
  assert!((is_nonnegative(0.)));
  assert!((!is_nonnegative(-1.)));
  assert!((!is_nonnegative(neg_infinity)));
  assert!((!is_nonnegative(1./neg_infinity)));
  assert!((!is_nonnegative(NaN)));
}

#[test]
pub fn test_to_str_inf() {
    assert!(to_str_digits(infinity, 10u) == ~"inf");
    assert!(to_str_digits(-infinity, 10u) == ~"-inf");
}

#[test]
pub fn test_round() {
    assert!(round(5.8) == 6.0);
    assert!(round(5.2) == 5.0);
    assert!(round(3.0) == 3.0);
    assert!(round(2.5) == 3.0);
    assert!(round(-3.5) == -4.0);
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

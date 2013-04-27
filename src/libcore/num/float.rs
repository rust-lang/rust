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

use from_str;
use libc::c_int;
use num::{Zero, One, strconv};
use prelude::*;

pub use f64::{add, sub, mul, quot, rem, lt, le, eq, ne, ge, gt};
pub use f64::logarithm;
pub use f64::{acos, asin, atan2, cbrt, ceil, copysign, cosh, floor};
pub use f64::{erf, erfc, exp, expm1, exp2, abs_sub};
pub use f64::{mul_add, fmax, fmin, next_after, frexp, hypot, ldexp};
pub use f64::{lgamma, ln, log_radix, ln1p, log10, log2, ilog_radix};
pub use f64::{modf, pow, powi, round, sinh, tanh, tgamma, trunc};
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

//
// Section: String Conversions
//

///
/// Converts a float to a string
///
/// # Arguments
///
/// * num - The float value
///
#[inline(always)]
pub fn to_str(num: float) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigAll);
    r
}

///
/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
///
#[inline(always)]
pub fn to_str_hex(num: float) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 16u, true, strconv::SignNeg, strconv::DigAll);
    r
}

///
/// Converts a float to a string in a given radix
///
/// # Arguments
///
/// * num - The float value
/// * radix - The base to use
///
/// # Failure
///
/// Fails if called on a special value like `inf`, `-inf` or `NaN` due to
/// possible misinterpretation of the result at higher bases. If those values
/// are expected, use `to_str_radix_special()` instead.
///
#[inline(always)]
pub fn to_str_radix(num: float, radix: uint) -> ~str {
    let (r, special) = strconv::to_str_common(
        &num, radix, true, strconv::SignNeg, strconv::DigAll);
    if special { fail!(~"number has a special value, \
                         try to_str_radix_special() if those are expected") }
    r
}

///
/// Converts a float to a string in a given radix, and a flag indicating
/// whether it's a special value
///
/// # Arguments
///
/// * num - The float value
/// * radix - The base to use
///
#[inline(always)]
pub fn to_str_radix_special(num: float, radix: uint) -> (~str, bool) {
    strconv::to_str_common(&num, radix, true,
                           strconv::SignNeg, strconv::DigAll)
}

///
/// Converts a float to a string with exactly the number of
/// provided significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
///
#[inline(always)]
pub fn to_str_exact(num: float, digits: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigExact(digits));
    r
}

///
/// Converts a float to a string with a maximum number of
/// significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
///
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

///
/// Convert a string in base 10 to a float.
/// Accepts a optional decimal exponent.
///
/// This function accepts strings such as
///
/// * '3.14'
/// * '+3.14', equivalent to '3.14'
/// * '-3.14'
/// * '2.5E10', or equivalently, '2.5e10'
/// * '2.5E-10'
/// * '.' (understood as 0)
/// * '5.'
/// * '.5', or, equivalently,  '0.5'
/// * '+inf', 'inf', '-inf', 'NaN'
///
/// Leading and trailing whitespace represent an error.
///
/// # Arguments
///
/// * num - A string
///
/// # Return value
///
/// `none` if the string did not represent a valid number.  Otherwise,
/// `Some(n)` where `n` is the floating-point number represented by `num`.
///
#[inline(always)]
pub fn from_str(num: &str) -> Option<float> {
    strconv::from_str_common(num, 10u, true, true, true,
                             strconv::ExpDec, false, false)
}

///
/// Convert a string in base 16 to a float.
/// Accepts a optional binary exponent.
///
/// This function accepts strings such as
///
/// * 'a4.fe'
/// * '+a4.fe', equivalent to 'a4.fe'
/// * '-a4.fe'
/// * '2b.aP128', or equivalently, '2b.ap128'
/// * '2b.aP-128'
/// * '.' (understood as 0)
/// * 'c.'
/// * '.c', or, equivalently,  '0.c'
/// * '+inf', 'inf', '-inf', 'NaN'
///
/// Leading and trailing whitespace represent an error.
///
/// # Arguments
///
/// * num - A string
///
/// # Return value
///
/// `none` if the string did not represent a valid number.  Otherwise,
/// `Some(n)` where `n` is the floating-point number represented by `[num]`.
///
#[inline(always)]
pub fn from_str_hex(num: &str) -> Option<float> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

///
/// Convert a string in an given base to a float.
///
/// Due to possible conflicts, this function does **not** accept
/// the special values `inf`, `-inf`, `+inf` and `NaN`, **nor**
/// does it recognize exponents of any kind.
///
/// Leading and trailing whitespace represent an error.
///
/// # Arguments
///
/// * num - A string
/// * radix - The base to use. Must lie in the range [2 .. 36]
///
/// # Return value
///
/// `none` if the string did not represent a valid number. Otherwise,
/// `Some(n)` where `n` is the floating-point number represented by `num`.
///
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

//
// Section: Arithmetics
//

///
/// Compute the exponentiation of an integer by another integer as a float
///
/// # Arguments
///
/// * x - The base
/// * pow - The exponent
///
/// # Return value
///
/// `NaN` if both `x` and `pow` are `0u`, otherwise `x^pow`
///
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
pub fn abs(x: float) -> float {
    f64::abs(x as f64) as float
}
#[inline(always)]
pub fn sqrt(x: float) -> float {
    f64::sqrt(x as f64) as float
}
#[inline(always)]
pub fn atan(x: float) -> float {
    f64::atan(x as f64) as float
}
#[inline(always)]
pub fn sin(x: float) -> float {
    f64::sin(x as f64) as float
}
#[inline(always)]
pub fn cos(x: float) -> float {
    f64::cos(x as f64) as float
}
#[inline(always)]
pub fn tan(x: float) -> float {
    f64::tan(x as f64) as float
}

impl Num for float {}

#[cfg(notest)]
impl Eq for float {
    #[inline(always)]
    fn eq(&self, other: &float) -> bool { (*self) == (*other) }
    #[inline(always)]
    fn ne(&self, other: &float) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl Ord for float {
    #[inline(always)]
    fn lt(&self, other: &float) -> bool { (*self) < (*other) }
    #[inline(always)]
    fn le(&self, other: &float) -> bool { (*self) <= (*other) }
    #[inline(always)]
    fn ge(&self, other: &float) -> bool { (*self) >= (*other) }
    #[inline(always)]
    fn gt(&self, other: &float) -> bool { (*self) > (*other) }
}

impl Orderable for float {
    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline(always)]
    fn min(&self, other: &float) -> float {
        (*self as f64).min(&(*other as f64)) as float
    }

    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline(always)]
    fn max(&self, other: &float) -> float {
        (*self as f64).max(&(*other as f64)) as float
    }

    /// Returns the number constrained within the range `mn <= self <= mx`.
    /// If any of the numbers are `NaN` then `NaN` is returned.
    #[inline(always)]
    fn clamp(&self, mn: &float, mx: &float) -> float {
        (*self as f64).clamp(&(*mn as f64), &(*mx as f64)) as float
    }
}

impl Zero for float {
    #[inline(always)]
    fn zero() -> float { 0.0 }

    /// Returns true if the number is equal to either `0.0` or `-0.0`
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0.0 || *self == -0.0 }
}

impl One for float {
    #[inline(always)]
    fn one() -> float { 1.0 }
}

impl Round for float {
    /// Round half-way cases toward `neg_infinity`
    #[inline(always)]
    fn floor(&self) -> float { floor(*self as f64) as float }

    /// Round half-way cases toward `infinity`
    #[inline(always)]
    fn ceil(&self) -> float { ceil(*self as f64) as float }

    /// Round half-way cases away from `0.0`
    #[inline(always)]
    fn round(&self) -> float { round(*self as f64) as float }

    /// The integer part of the number (rounds towards `0.0`)
    #[inline(always)]
    fn trunc(&self) -> float { trunc(*self as f64) as float }

    ///
    /// The fractional part of the number, satisfying:
    ///
    /// ~~~
    /// assert!(x == trunc(x) + fract(x))
    /// ~~~
    ///
    #[inline(always)]
    fn fract(&self) -> float { *self - self.trunc() }
}

impl Fractional for float {
    /// The reciprocal (multiplicative inverse) of the number
    #[inline(always)]
    fn recip(&self) -> float { 1.0 / *self }
}

impl Real for float {
    /// Archimedes' constant
    #[inline(always)]
    fn pi() -> float { 3.14159265358979323846264338327950288 }

    /// 2.0 * pi
    #[inline(always)]
    fn two_pi() -> float { 6.28318530717958647692528676655900576 }

    /// pi / 2.0
    #[inline(always)]
    fn frac_pi_2() -> float { 1.57079632679489661923132169163975144 }

    /// pi / 3.0
    #[inline(always)]
    fn frac_pi_3() -> float { 1.04719755119659774615421446109316763 }

    /// pi / 4.0
    #[inline(always)]
    fn frac_pi_4() -> float { 0.785398163397448309615660845819875721 }

    /// pi / 6.0
    #[inline(always)]
    fn frac_pi_6() -> float { 0.52359877559829887307710723054658381 }

    /// pi / 8.0
    #[inline(always)]
    fn frac_pi_8() -> float { 0.39269908169872415480783042290993786 }

    /// 1.0 / pi
    #[inline(always)]
    fn frac_1_pi() -> float { 0.318309886183790671537767526745028724 }

    /// 2.0 / pi
    #[inline(always)]
    fn frac_2_pi() -> float { 0.636619772367581343075535053490057448 }

    /// 2 .0/ sqrt(pi)
    #[inline(always)]
    fn frac_2_sqrtpi() -> float { 1.12837916709551257389615890312154517 }

    /// sqrt(2.0)
    #[inline(always)]
    fn sqrt2() -> float { 1.41421356237309504880168872420969808 }

    /// 1.0 / sqrt(2.0)
    #[inline(always)]
    fn frac_1_sqrt2() -> float { 0.707106781186547524400844362104849039 }

    /// Euler's number
    #[inline(always)]
    fn e() -> float { 2.71828182845904523536028747135266250 }

    /// log2(e)
    #[inline(always)]
    fn log2_e() -> float { 1.44269504088896340735992468100189214 }

    /// log10(e)
    #[inline(always)]
    fn log10_e() -> float { 0.434294481903251827651128918916605082 }

    /// log(2.0)
    #[inline(always)]
    fn log_2() -> float { 0.693147180559945309417232121458176568 }

    /// log(10.0)
    #[inline(always)]
    fn log_10() -> float { 2.30258509299404568401799145468436421 }

    #[inline(always)]
    fn pow(&self, n: float) -> float { pow(*self as f64, n as f64) as float }

    #[inline(always)]
    fn exp(&self) -> float { exp(*self as f64) as float }

    #[inline(always)]
    fn exp2(&self) -> float { exp2(*self as f64) as float }

    #[inline(always)]
    fn expm1(&self) -> float { expm1(*self as f64) as float }

    #[inline(always)]
    fn ldexp(&self, n: int) -> float { ldexp(*self as f64, n as c_int) as float }

    #[inline(always)]
    fn log(&self) -> float { ln(*self as f64) as float }

    #[inline(always)]
    fn log2(&self) -> float { log2(*self as f64) as float }

    #[inline(always)]
    fn log10(&self) -> float { log10(*self as f64) as float }

    #[inline(always)]
    fn log_radix(&self) -> float { log_radix(*self as f64) as float }

    #[inline(always)]
    fn ilog_radix(&self) -> int { ilog_radix(*self as f64) as int }

    #[inline(always)]
    fn sqrt(&self) -> float { sqrt(*self) }

    #[inline(always)]
    fn rsqrt(&self) -> float { self.sqrt().recip() }

    #[inline(always)]
    fn cbrt(&self) -> float { cbrt(*self as f64) as float }

    /// Converts to degrees, assuming the number is in radians
    #[inline(always)]
    fn to_degrees(&self) -> float { *self * (180.0 / Real::pi::<float>()) }

    /// Converts to radians, assuming the number is in degrees
    #[inline(always)]
    fn to_radians(&self) -> float { *self * (Real::pi::<float>() / 180.0) }

    #[inline(always)]
    fn hypot(&self, other: float) -> float { hypot(*self as f64, other as f64) as float }

    #[inline(always)]
    fn sin(&self) -> float { sin(*self) }

    #[inline(always)]
    fn cos(&self) -> float { cos(*self) }

    #[inline(always)]
    fn tan(&self) -> float { tan(*self) }

    #[inline(always)]
    fn asin(&self) -> float { asin(*self as f64) as float }

    #[inline(always)]
    fn acos(&self) -> float { acos(*self as f64) as float }

    #[inline(always)]
    fn atan(&self) -> float { atan(*self) }

    #[inline(always)]
    fn atan2(&self, other: float) -> float { atan2(*self as f64, other as f64) as float }

    #[inline(always)]
    fn sinh(&self) -> float { sinh(*self as f64) as float }

    #[inline(always)]
    fn cosh(&self) -> float { cosh(*self as f64) as float }

    #[inline(always)]
    fn tanh(&self) -> float { tanh(*self as f64) as float }
}

impl RealExt for float {
    #[inline(always)]
    fn lgamma(&self) -> (int, float) {
        let mut sign = 0;
        let result = lgamma(*self as f64, &mut sign);
        (sign as int, result as float)
    }

    #[inline(always)]
    fn tgamma(&self) -> float { tgamma(*self as f64) as float }

    #[inline(always)]
    fn j0(&self) -> float { j0(*self as f64) as float }

    #[inline(always)]
    fn j1(&self) -> float { j1(*self as f64) as float }

    #[inline(always)]
    fn jn(&self, n: int) -> float { jn(n as c_int, *self as f64) as float }

    #[inline(always)]
    fn y0(&self) -> float { y0(*self as f64) as float }

    #[inline(always)]
    fn y1(&self) -> float { y1(*self as f64) as float }

    #[inline(always)]
    fn yn(&self, n: int) -> float { yn(n as c_int, *self as f64) as float }
}

#[cfg(notest)]
impl Add<float,float> for float {
    #[inline(always)]
    fn add(&self, other: &float) -> float { *self + *other }
}

#[cfg(notest)]
impl Sub<float,float> for float {
    #[inline(always)]
    fn sub(&self, other: &float) -> float { *self - *other }
}

#[cfg(notest)]
impl Mul<float,float> for float {
    #[inline(always)]
    fn mul(&self, other: &float) -> float { *self * *other }
}

#[cfg(stage0,notest)]
impl Div<float,float> for float {
    #[inline(always)]
    fn div(&self, other: &float) -> float { *self / *other }
}
#[cfg(not(stage0),notest)]
impl Quot<float,float> for float {
    #[inline(always)]
    fn quot(&self, other: &float) -> float { *self / *other }
}
#[cfg(stage0,notest)]
impl Modulo<float,float> for float {
    #[inline(always)]
    fn modulo(&self, other: &float) -> float { *self % *other }
}
#[cfg(not(stage0),notest)]
impl Rem<float,float> for float {
    #[inline(always)]
    fn rem(&self, other: &float) -> float { *self % *other }
}
#[cfg(notest)]
impl Neg<float> for float {
    #[inline(always)]
    fn neg(&self) -> float { -*self }
}

impl Signed for float {
    /// Computes the absolute value. Returns `NaN` if the number is `NaN`.
    #[inline(always)]
    fn abs(&self) -> float { abs(*self) }

    ///
    /// # Returns
    ///
    /// - `1.0` if the number is positive, `+0.0` or `infinity`
    /// - `-1.0` if the number is negative, `-0.0` or `neg_infinity`
    /// - `NaN` if the number is NaN
    ///
    #[inline(always)]
    fn signum(&self) -> float {
        if self.is_NaN() { NaN } else { f64::copysign(1.0, *self as f64) as float }
    }

    /// Returns `true` if the number is positive, including `+0.0` and `infinity`
    #[inline(always)]
    fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == infinity }

    /// Returns `true` if the number is negative, including `-0.0` and `neg_infinity`
    #[inline(always)]
    fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == neg_infinity }
}

impl Bounded for float {
    #[inline(always)]
    fn min_value() -> float { Bounded::min_value::<f64>() as float }

    #[inline(always)]
    fn max_value() -> float { Bounded::max_value::<f64>() as float }
}

impl Primitive for float {
    #[inline(always)]
    fn bits() -> uint { Primitive::bits::<f64>() }

    #[inline(always)]
    fn bytes() -> uint { Primitive::bytes::<f64>() }
}

impl Float for float {
    #[inline(always)]
    fn NaN() -> float { 0.0 / 0.0 }

    #[inline(always)]
    fn infinity() -> float { 1.0 / 0.0 }

    #[inline(always)]
    fn neg_infinity() -> float { -1.0 / 0.0 }

    #[inline(always)]
    fn neg_zero() -> float { -0.0 }

    #[inline(always)]
    fn is_NaN(&self) -> bool { *self != *self }

    #[inline(always)]
    fn mantissa_digits() -> uint { Float::mantissa_digits::<f64>() }

    #[inline(always)]
    fn digits() -> uint { Float::digits::<f64>() }

    #[inline(always)]
    fn epsilon() -> float { Float::epsilon::<f64>() as float }

    #[inline(always)]
    fn min_exp() -> int { Float::min_exp::<f64>() }

    #[inline(always)]
    fn max_exp() -> int { Float::max_exp::<f64>() }

    #[inline(always)]
    fn min_10_exp() -> int { Float::min_10_exp::<f64>() }

    #[inline(always)]
    fn max_10_exp() -> int { Float::max_10_exp::<f64>() }

    /// Returns `true` if the number is infinite
    #[inline(always)]
    fn is_infinite(&self) -> bool {
        *self == Float::infinity() || *self == Float::neg_infinity()
    }

    /// Returns `true` if the number is finite
    #[inline(always)]
    fn is_finite(&self) -> bool {
        !(self.is_NaN() || self.is_infinite())
    }

    ///
    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error. This
    /// produces a more accurate result with better performance than a separate multiplication
    /// operation followed by an add.
    ///
    #[inline(always)]
    fn mul_add(&self, a: float, b: float) -> float {
        mul_add(*self as f64, a as f64, b as f64) as float
    }

    /// Returns the next representable floating-point value in the direction of `other`
    #[inline(always)]
    fn next_after(&self, other: float) -> float {
        next_after(*self as f64, other as f64) as float
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    macro_rules! assert_fuzzy_eq(
        ($a:expr, $b:expr) => ({
            let a = $a, b = $b;
            if !((a - b).abs() < 1.0e-6) {
                fail!(fmt!("The values were not approximately equal. Found: %? and %?", a, b));
            }
        })
    )

    #[test]
    fn test_num() {
        num::test_num(10f, 2f);
    }

    #[test]
    fn test_min() {
        assert_eq!(1f.min(&2f), 1f);
        assert_eq!(2f.min(&1f), 1f);
    }

    #[test]
    fn test_max() {
        assert_eq!(1f.max(&2f), 2f);
        assert_eq!(2f.max(&1f), 2f);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(1f.clamp(&2f, &4f), 2f);
        assert_eq!(8f.clamp(&2f, &4f), 4f);
        assert_eq!(3f.clamp(&2f, &4f), 3f);
        assert!(3f.clamp(&Float::NaN::<float>(), &4f).is_NaN());
        assert!(3f.clamp(&2f, &Float::NaN::<float>()).is_NaN());
        assert!(Float::NaN::<float>().clamp(&2f, &4f).is_NaN());
    }

    #[test]
    fn test_floor() {
        assert_fuzzy_eq!(1.0f.floor(), 1.0f);
        assert_fuzzy_eq!(1.3f.floor(), 1.0f);
        assert_fuzzy_eq!(1.5f.floor(), 1.0f);
        assert_fuzzy_eq!(1.7f.floor(), 1.0f);
        assert_fuzzy_eq!(0.0f.floor(), 0.0f);
        assert_fuzzy_eq!((-0.0f).floor(), -0.0f);
        assert_fuzzy_eq!((-1.0f).floor(), -1.0f);
        assert_fuzzy_eq!((-1.3f).floor(), -2.0f);
        assert_fuzzy_eq!((-1.5f).floor(), -2.0f);
        assert_fuzzy_eq!((-1.7f).floor(), -2.0f);
    }

    #[test]
    fn test_ceil() {
        assert_fuzzy_eq!(1.0f.ceil(), 1.0f);
        assert_fuzzy_eq!(1.3f.ceil(), 2.0f);
        assert_fuzzy_eq!(1.5f.ceil(), 2.0f);
        assert_fuzzy_eq!(1.7f.ceil(), 2.0f);
        assert_fuzzy_eq!(0.0f.ceil(), 0.0f);
        assert_fuzzy_eq!((-0.0f).ceil(), -0.0f);
        assert_fuzzy_eq!((-1.0f).ceil(), -1.0f);
        assert_fuzzy_eq!((-1.3f).ceil(), -1.0f);
        assert_fuzzy_eq!((-1.5f).ceil(), -1.0f);
        assert_fuzzy_eq!((-1.7f).ceil(), -1.0f);
    }

    #[test]
    fn test_round() {
        assert_fuzzy_eq!(1.0f.round(), 1.0f);
        assert_fuzzy_eq!(1.3f.round(), 1.0f);
        assert_fuzzy_eq!(1.5f.round(), 2.0f);
        assert_fuzzy_eq!(1.7f.round(), 2.0f);
        assert_fuzzy_eq!(0.0f.round(), 0.0f);
        assert_fuzzy_eq!((-0.0f).round(), -0.0f);
        assert_fuzzy_eq!((-1.0f).round(), -1.0f);
        assert_fuzzy_eq!((-1.3f).round(), -1.0f);
        assert_fuzzy_eq!((-1.5f).round(), -2.0f);
        assert_fuzzy_eq!((-1.7f).round(), -2.0f);
    }

    #[test]
    fn test_trunc() {
        assert_fuzzy_eq!(1.0f.trunc(), 1.0f);
        assert_fuzzy_eq!(1.3f.trunc(), 1.0f);
        assert_fuzzy_eq!(1.5f.trunc(), 1.0f);
        assert_fuzzy_eq!(1.7f.trunc(), 1.0f);
        assert_fuzzy_eq!(0.0f.trunc(), 0.0f);
        assert_fuzzy_eq!((-0.0f).trunc(), -0.0f);
        assert_fuzzy_eq!((-1.0f).trunc(), -1.0f);
        assert_fuzzy_eq!((-1.3f).trunc(), -1.0f);
        assert_fuzzy_eq!((-1.5f).trunc(), -1.0f);
        assert_fuzzy_eq!((-1.7f).trunc(), -1.0f);
    }

    #[test]
    fn test_fract() {
        assert_fuzzy_eq!(1.0f.fract(), 0.0f);
        assert_fuzzy_eq!(1.3f.fract(), 0.3f);
        assert_fuzzy_eq!(1.5f.fract(), 0.5f);
        assert_fuzzy_eq!(1.7f.fract(), 0.7f);
        assert_fuzzy_eq!(0.0f.fract(), 0.0f);
        assert_fuzzy_eq!((-0.0f).fract(), -0.0f);
        assert_fuzzy_eq!((-1.0f).fract(), -0.0f);
        assert_fuzzy_eq!((-1.3f).fract(), -0.3f);
        assert_fuzzy_eq!((-1.5f).fract(), -0.5f);
        assert_fuzzy_eq!((-1.7f).fract(), -0.7f);
    }

    #[test]
    fn test_real_consts() {
        assert_fuzzy_eq!(Real::two_pi::<float>(), 2f * Real::pi::<float>());
        assert_fuzzy_eq!(Real::frac_pi_2::<float>(), Real::pi::<float>() / 2f);
        assert_fuzzy_eq!(Real::frac_pi_3::<float>(), Real::pi::<float>() / 3f);
        assert_fuzzy_eq!(Real::frac_pi_4::<float>(), Real::pi::<float>() / 4f);
        assert_fuzzy_eq!(Real::frac_pi_6::<float>(), Real::pi::<float>() / 6f);
        assert_fuzzy_eq!(Real::frac_pi_8::<float>(), Real::pi::<float>() / 8f);
        assert_fuzzy_eq!(Real::frac_1_pi::<float>(), 1f / Real::pi::<float>());
        assert_fuzzy_eq!(Real::frac_2_pi::<float>(), 2f / Real::pi::<float>());
        assert_fuzzy_eq!(Real::frac_2_sqrtpi::<float>(), 2f / Real::pi::<float>().sqrt());
        assert_fuzzy_eq!(Real::sqrt2::<float>(), 2f.sqrt());
        assert_fuzzy_eq!(Real::frac_1_sqrt2::<float>(), 1f / 2f.sqrt());
        assert_fuzzy_eq!(Real::log2_e::<float>(), Real::e::<float>().log2());
        assert_fuzzy_eq!(Real::log10_e::<float>(), Real::e::<float>().log10());
        assert_fuzzy_eq!(Real::log_2::<float>(), 2f.log());
        assert_fuzzy_eq!(Real::log_10::<float>(), 10f.log());
    }

    #[test]
    fn test_signed() {
        assert_eq!(infinity.abs(), infinity);
        assert_eq!(1f.abs(), 1f);
        assert_eq!(0f.abs(), 0f);
        assert_eq!((-0f).abs(), 0f);
        assert_eq!((-1f).abs(), 1f);
        assert_eq!(neg_infinity.abs(), infinity);
        assert_eq!((1f/neg_infinity).abs(), 0f);
        assert!(NaN.abs().is_NaN());

        assert_eq!(infinity.signum(), 1f);
        assert_eq!(1f.signum(), 1f);
        assert_eq!(0f.signum(), 1f);
        assert_eq!((-0f).signum(), -1f);
        assert_eq!((-1f).signum(), -1f);
        assert_eq!(neg_infinity.signum(), -1f);
        assert_eq!((1f/neg_infinity).signum(), -1f);
        assert!(NaN.signum().is_NaN());

        assert!(infinity.is_positive());
        assert!(1f.is_positive());
        assert!(0f.is_positive());
        assert!(!(-0f).is_positive());
        assert!(!(-1f).is_positive());
        assert!(!neg_infinity.is_positive());
        assert!(!(1f/neg_infinity).is_positive());
        assert!(!NaN.is_positive());

        assert!(!infinity.is_negative());
        assert!(!1f.is_negative());
        assert!(!0f.is_negative());
        assert!((-0f).is_negative());
        assert!((-1f).is_negative());
        assert!(neg_infinity.is_negative());
        assert!((1f/neg_infinity).is_negative());
        assert!(!NaN.is_negative());
    }

    #[test]
    fn test_primitive() {
        assert_eq!(Primitive::bits::<float>(), sys::size_of::<float>() * 8);
        assert_eq!(Primitive::bytes::<float>(), sys::size_of::<float>());
    }

    #[test]
    pub fn test_to_str_exact_do_decimal() {
        let s = to_str_exact(5.0, 4u);
        assert_eq!(s, ~"5.0000");
    }

    #[test]
    pub fn test_from_str() {
        assert_eq!(from_str(~"3"), Some(3.));
        assert_eq!(from_str(~"3.14"), Some(3.14));
        assert_eq!(from_str(~"+3.14"), Some(3.14));
        assert_eq!(from_str(~"-3.14"), Some(-3.14));
        assert_eq!(from_str(~"2.5E10"), Some(25000000000.));
        assert_eq!(from_str(~"2.5e10"), Some(25000000000.));
        assert_eq!(from_str(~"25000000000.E-10"), Some(2.5));
        assert_eq!(from_str(~"."), Some(0.));
        assert_eq!(from_str(~".e1"), Some(0.));
        assert_eq!(from_str(~".e-1"), Some(0.));
        assert_eq!(from_str(~"5."), Some(5.));
        assert_eq!(from_str(~".5"), Some(0.5));
        assert_eq!(from_str(~"0.5"), Some(0.5));
        assert_eq!(from_str(~"-.5"), Some(-0.5));
        assert_eq!(from_str(~"-5"), Some(-5.));
        assert_eq!(from_str(~"inf"), Some(infinity));
        assert_eq!(from_str(~"+inf"), Some(infinity));
        assert_eq!(from_str(~"-inf"), Some(neg_infinity));
        // note: NaN != NaN, hence this slightly complex test
        match from_str(~"NaN") {
            Some(f) => assert!(f.is_NaN()),
            None => fail!()
        }
        // note: -0 == 0, hence these slightly more complex tests
        match from_str(~"-0") {
            Some(v) if v.is_zero() => assert!(v.is_negative()),
            _ => fail!()
        }
        match from_str(~"0") {
            Some(v) if v.is_zero() => assert!(v.is_positive()),
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
        assert_eq!(from_str_hex(~"a4"), Some(164.));
        assert_eq!(from_str_hex(~"a4.fe"), Some(164.9921875));
        assert_eq!(from_str_hex(~"-a4.fe"), Some(-164.9921875));
        assert_eq!(from_str_hex(~"+a4.fe"), Some(164.9921875));
        assert_eq!(from_str_hex(~"ff0P4"), Some(0xff00 as float));
        assert_eq!(from_str_hex(~"ff0p4"), Some(0xff00 as float));
        assert_eq!(from_str_hex(~"ff0p-4"), Some(0xff as float));
        assert_eq!(from_str_hex(~"."), Some(0.));
        assert_eq!(from_str_hex(~".p1"), Some(0.));
        assert_eq!(from_str_hex(~".p-1"), Some(0.));
        assert_eq!(from_str_hex(~"f."), Some(15.));
        assert_eq!(from_str_hex(~".f"), Some(0.9375));
        assert_eq!(from_str_hex(~"0.f"), Some(0.9375));
        assert_eq!(from_str_hex(~"-.f"), Some(-0.9375));
        assert_eq!(from_str_hex(~"-f"), Some(-15.));
        assert_eq!(from_str_hex(~"inf"), Some(infinity));
        assert_eq!(from_str_hex(~"+inf"), Some(infinity));
        assert_eq!(from_str_hex(~"-inf"), Some(neg_infinity));
        // note: NaN != NaN, hence this slightly complex test
        match from_str_hex(~"NaN") {
            Some(f) => assert!(f.is_NaN()),
            None => fail!()
        }
        // note: -0 == 0, hence these slightly more complex tests
        match from_str_hex(~"-0") {
            Some(v) if v.is_zero() => assert!(v.is_negative()),
            _ => fail!()
        }
        match from_str_hex(~"0") {
            Some(v) if v.is_zero() => assert!(v.is_positive()),
            _ => fail!()
        }
        assert_eq!(from_str_hex(~"e"), Some(14.));
        assert_eq!(from_str_hex(~"E"), Some(14.));
        assert_eq!(from_str_hex(~"E1"), Some(225.));
        assert_eq!(from_str_hex(~"1e1e1"), Some(123361.));
        assert_eq!(from_str_hex(~"1e1.1"), Some(481.0625));

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
        assert_eq!(to_str_hex(164.), ~"a4");
        assert_eq!(to_str_hex(164.9921875), ~"a4.fe");
        assert_eq!(to_str_hex(-164.9921875), ~"-a4.fe");
        assert_eq!(to_str_hex(0xff00 as float), ~"ff00");
        assert_eq!(to_str_hex(-(0xff00 as float)), ~"-ff00");
        assert_eq!(to_str_hex(0.), ~"0");
        assert_eq!(to_str_hex(15.), ~"f");
        assert_eq!(to_str_hex(-15.), ~"-f");
        assert_eq!(to_str_hex(0.9375), ~"0.f");
        assert_eq!(to_str_hex(-0.9375), ~"-0.f");
        assert_eq!(to_str_hex(infinity), ~"inf");
        assert_eq!(to_str_hex(neg_infinity), ~"-inf");
        assert_eq!(to_str_hex(NaN), ~"NaN");
        assert_eq!(to_str_hex(0.), ~"0");
        assert_eq!(to_str_hex(-0.), ~"-0");
    }

    #[test]
    pub fn test_to_str_radix() {
        assert_eq!(to_str_radix(36., 36u), ~"10");
        assert_eq!(to_str_radix(8.125, 2u), ~"1000.001");
    }

    #[test]
    pub fn test_from_str_radix() {
        assert_eq!(from_str_radix(~"10", 36u), Some(36.));
        assert_eq!(from_str_radix(~"1000.001", 2u), Some(8.125));
    }

    #[test]
    pub fn test_to_str_inf() {
        assert_eq!(to_str_digits(infinity, 10u), ~"inf");
        assert_eq!(to_str_digits(-infinity, 10u), ~"-inf");
    }
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

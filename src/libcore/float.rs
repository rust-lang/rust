/*
Module: float
*/

// Currently this module supports from -lm
// C95 + log2 + log1p + trunc + round + rint

export t;

export consts;

export
    acos, asin, atan, atan2, ceil, cos, cosh, exp, abs, floor, fmod, frexp,
    ldexp, ln, ln1p, log10, log2, modf, rint, round, pow, sin, sinh, sqrt,
    tan, tanh, trunc;

export to_str_common, to_str_exact, to_str, from_str;
export lt, le, eq, ne, gt, eq;
export NaN, isNaN, infinity, neg_infinity;
export pow_uint_to_uint_as_float;
export min, max;
export add, sub, mul, div;
export positive, negative, nonpositive, nonnegative;

import mtypes::m_float;
import ctypes::c_int;
import ptr;

// PORT This must match in width according to architecture
import f64;
import m_float = f64;

type t = m_float;

/**
 * Section: String Conversions
 */

/*
Function: to_str_common

Converts a float to a string

Parameters:

num - The float value
digits - The number of significant digits
exact - Whether to enforce the exact number of significant digits
*/
fn to_str_common(num: float, digits: uint, exact: bool) -> str {
    let (num, accum) = num < 0.0 ? (-num, "-") : (num, "");
    let trunc = num as uint;
    let frac = num - (trunc as float);
    accum += uint::str(trunc);
    if frac == 0.0 || digits == 0u { ret accum; }
    accum += ".";
    let i = digits;
    let epsilon = 1. / pow_uint_to_uint_as_float(10u, i);
    while i > 0u && (frac >= epsilon || exact) {
        frac *= 10.0;
        epsilon *= 10.0;
        let digit = frac as uint;
        accum += uint::str(digit);
        frac -= digit as float;
        i -= 1u;
    }
    ret accum;

}

/*
Function: to_str

Converts a float to a string with exactly the number of provided significant
digits

Parameters:

num - The float value
digits - The number of significant digits
*/
fn to_str_exact(num: float, digits: uint) -> str {
    to_str_common(num, digits, true)
}

/*
Function: to_str

Converts a float to a string with a maximum number of significant digits

Parameters:

num - The float value
digits - The number of significant digits
*/
fn to_str(num: float, digits: uint) -> str {
    to_str_common(num, digits, false)
}

/*
Function: from_str

Convert a string to a float

This function accepts strings such as
* "3.14"
* "+3.14", equivalent to "3.14"
* "-3.14"
* "2.5E10", or equivalently, "2.5e10"
* "2.5E-10"
* "", or, equivalently, "." (understood as 0)
* "5."
* ".5", or, equivalently,  "0.5"

Leading and trailing whitespace are ignored.

Parameters:

num - A string, possibly empty.

Returns:

<NaN> If the string did not represent a valid number.
Otherwise, the floating-point number represented [num].
*/
fn from_str(num: str) -> float {
   let num = str::trim(num);

   let pos = 0u;                  //Current byte position in the string.
                                  //Used to walk the string in O(n).
   let len = str::byte_len(num);  //Length of the string, in bytes.

   if len == 0u { ret 0.; }
   let total = 0f;                //Accumulated result
   let c     = 'z';               //Latest char.

   //The string must start with one of the following characters.
   alt str::char_at(num, 0u) {
      '-' | '+' | '0' to '9' | '.' {}
      _ { ret NaN; }
   }

   //Determine if first char is '-'/'+'. Set [pos] and [neg] accordingly.
   let neg = false;               //Sign of the result
   alt str::char_at(num, 0u) {
      '-' {
          neg = true;
          pos = 1u;
      }
      '+' {
          pos = 1u;
      }
      _ {}
   }

   //Examine the following chars until '.', 'e', 'E'
   while(pos < len) {
       let char_range = str::char_range_at(num, pos);
       c   = char_range.ch;
       pos = char_range.next;
       alt c {
         '0' to '9' {
           total = total * 10f;
           total += ((c as int) - ('0' as int)) as float;
         }
         '.' | 'e' | 'E' {
           break;
         }
         _ {
           ret NaN;
         }
       }
   }

   if c == '.' {//Examine decimal part
      let decimal = 1f;
      while(pos < len) {
         let char_range = str::char_range_at(num, pos);
         c = char_range.ch;
         pos = char_range.next;
         alt c {
            '0' | '1' | '2' | '3' | '4' | '5' | '6'| '7' | '8' | '9'  {
                 decimal /= 10f;
                 total += (((c as int) - ('0' as int)) as float)*decimal;
             }
             'e' | 'E' {
                 break;
             }
             _ {
                 ret NaN;
             }
         }
      }
   }

   if (c == 'e') | (c == 'E') {//Examine exponent
      let exponent = 0u;
      let neg_exponent = false;
      if(pos < len) {
          let char_range = str::char_range_at(num, pos);
          c   = char_range.ch;
          alt c  {
             '+' {
                pos = char_range.next;
             }
             '-' {
                pos = char_range.next;
                neg_exponent = true;
             }
             _ {}
          }
          while(pos < len) {
             let char_range = str::char_range_at(num, pos);
             c = char_range.ch;
             alt c {
                 '0' | '1' | '2' | '3' | '4' | '5' | '6'| '7' | '8' | '9' {
                     exponent *= 10u;
                     exponent += ((c as uint) - ('0' as uint));
                 }
                 _ {
                     break;
                 }
             }
             pos = char_range.next;
          }
          let multiplier = pow_uint_to_uint_as_float(10u, exponent);
              //Note: not [int::pow], otherwise, we'll quickly
              //end up with a nice overflow
          if neg_exponent {
             total = total / multiplier;
          } else {
             total = total * multiplier;
          }
      } else {
         ret NaN;
      }
   }

   if(pos < len) {
     ret NaN;
   } else {
     if(neg) {
        total *= -1f;
     }
     ret total;
   }
}

/**
 * Section: Arithmetics
 */

/*
Function: pow_uint_to_uint_as_float

Compute the exponentiation of an integer by another integer as a float.

Parameters:
x - The base.
pow - The exponent.

Returns:
<NaN> of both `x` and `pow` are `0u`, otherwise `x^pow`.
*/
fn pow_uint_to_uint_as_float(x: uint, pow: uint) -> float {
   if x == 0u {
      if pow == 0u {
        ret NaN;
      }
       ret 0.;
   }
   let my_pow     = pow;
   let total      = 1f;
   let multiplier = x as float;
   while (my_pow > 0u) {
     if my_pow % 2u == 1u {
       total = total * multiplier;
     }
     my_pow     /= 2u;
     multiplier *= multiplier;
   }
   ret total;
}


/* Const: NaN */
const NaN: float = 0./0.;

/* Const: infinity */
const infinity: float = 1./0.;

/* Const: neg_infinity */
const neg_infinity: float = -1./0.;

/* Predicate: isNaN */
pure fn isNaN(f: float) -> bool { f != f }

/* Function: add */
pure fn add(x: float, y: float) -> float { ret x + y; }

/* Function: sub */
pure fn sub(x: float, y: float) -> float { ret x - y; }

/* Function: mul */
pure fn mul(x: float, y: float) -> float { ret x * y; }

/* Function: div */
pure fn div(x: float, y: float) -> float { ret x / y; }

/* Function: rem */
pure fn rem(x: float, y: float) -> float { ret x % y; }

/* Predicate: lt */
pure fn lt(x: float, y: float) -> bool { ret x < y; }

/* Predicate: le */
pure fn le(x: float, y: float) -> bool { ret x <= y; }

/* Predicate: eq */
pure fn eq(x: float, y: float) -> bool { ret x == y; }

/* Predicate: ne */
pure fn ne(x: float, y: float) -> bool { ret x != y; }

/* Predicate: ge */
pure fn ge(x: float, y: float) -> bool { ret x >= y; }

/* Predicate: gt */
pure fn gt(x: float, y: float) -> bool { ret x > y; }

/*
Predicate: positive

Returns true if `x` is a positive number, including +0.0 and +Infinity.
 */
pure fn positive(x: float) -> bool { ret x > 0. || (1./x) == infinity; }

/*
Predicate: negative

Returns true if `x` is a negative number, including -0.0 and -Infinity.
 */
pure fn negative(x: float) -> bool { ret x < 0. || (1./x) == neg_infinity; }

/*
Predicate: nonpositive

Returns true if `x` is a negative number, including -0.0 and -Infinity.
(This is the same as `float::negative`.)
*/
pure fn nonpositive(x: float) -> bool {
  ret x < 0. || (1./x) == neg_infinity;
}

/*
Predicate: nonnegative

Returns true if `x` is a positive number, including +0.0 and +Infinity.
(This is the same as `float::positive`.)
*/
pure fn nonnegative(x: float) -> bool {
  ret x > 0. || (1./x) == infinity;
}

/*
Module: consts
*/
mod consts {
    /*
    Const: pi

    Archimedes' constant
    */
    const pi: float = 3.14159265358979323846264338327950288;

    /*
    Const: frac_pi_2

    pi/2.0
    */
    const frac_pi_2: float = 1.57079632679489661923132169163975144;

    /*
    Const: frac_pi_4

    pi/4.0
    */
    const frac_pi_4: float = 0.785398163397448309615660845819875721;

    /*
    Const: frac_1_pi

    1.0/pi
    */
    const frac_1_pi: float = 0.318309886183790671537767526745028724;

    /*
    Const: frac_2_pi

    2.0/pi
    */
    const frac_2_pi: float = 0.636619772367581343075535053490057448;

    /*
    Const: frac_2_sqrtpi

    2.0/sqrt(pi)
    */
    const frac_2_sqrtpi: float = 1.12837916709551257389615890312154517;

    /*
    Const: sqrt2

    sqrt(2.0)
    */
    const sqrt2: float = 1.41421356237309504880168872420969808;

    /*
    Const: frac_1_sqrt2

    1.0/sqrt(2.0)
    */
    const frac_1_sqrt2: float = 0.707106781186547524400844362104849039;

    /*
    Const: e

    Euler's number
    */
    const e: float = 2.71828182845904523536028747135266250;

    /*
    Const: log2_e

    log2(e)
    */
    const log2_e: float = 1.44269504088896340735992468100189214;

    /*
    Const: log10_e

    log10(e)
    */
    const log10_e: float = 0.434294481903251827651128918916605082;

    /*
    Const: ln_2

    ln(2.0)
    */
    const ln_2: float = 0.693147180559945309417232121458176568;

    /*
    Const: ln_10

    ln(10.0)
    */
    const ln_10: float = 2.30258509299404568401799145468436421;
}


// FIXME min/max type specialize via libm when overloading works
// (in theory fmax/fmin, fmaxf, fminf /should/ be faster)

/*
Function: min

Returns the minimum of two values
*/
pure fn min<copy T>(x: T, y: T) -> T { x < y ? x : y }

/*
Function: max

Returns the maximum of two values
*/
pure fn max<copy T>(x: T, y: T) -> T { x < y ? y : x }

/*
Function: acos

Returns the arccosine of an angle (measured in rad)
*/
pure fn acos(x: float) -> float
    { ret m_float::acos(x as m_float) as float }

/*
Function: asin

Returns the arcsine of an angle (measured in rad)
*/
pure fn asin(x: float) -> float
    { ret m_float::asin(x as m_float) as float }

/*
Function: atan

Returns the arctangents of an angle (measured in rad)
*/
pure fn atan(x: float) -> float
    { ret m_float::atan(x as m_float) as float }


/*
Function: atan2

Returns the arctangent of an angle (measured in rad)
*/
pure fn atan2(y: float, x: float) -> float
    { ret m_float::atan2(y as m_float, x as m_float) as float }

/*
Function: ceil

Returns the smallest integral value less than or equal to `n`
*/
pure fn ceil(n: float) -> float
    { ret m_float::ceil(n as m_float) as float }

/*
Function: cos

Returns the cosine of an angle `x` (measured in rad)
*/
pure fn cos(x: float) -> float
    { ret m_float::cos(x as m_float) as float }

/*
Function: cosh

Returns the hyperbolic cosine of `x`

*/
pure fn cosh(x: float) -> float
    { ret m_float::cosh(x as m_float) as float }


/*
Function: exp

Returns `consts::e` to the power of `n*
*/
pure fn exp(n: float) -> float
    { ret m_float::exp(n as m_float) as float }

/*
Function: abs

Returns the absolute value of  `n`
*/
pure fn abs(n: float) -> float
    { ret m_float::abs(n as m_float) as float }

/*
Function: floor

Returns the largest integral value less than or equal to `n`
*/
pure fn floor(n: float) -> float
    { ret m_float::floor(n as m_float) as float }

/*
Function: fmod

Returns the floating-point remainder of `x/y`
*/
pure fn fmod(x: float, y: float) -> float
    { ret m_float::fmod(x as m_float, y as m_float) as float }

/*
Function: ln

Returns the natural logaritm of `n`
*/
pure fn ln(n: float) -> float
    { ret m_float::ln(n as m_float) as float }

/*
Function: ldexp

Returns `x` multiplied by 2 to the power of `n`
*/
pure fn ldexp(n: float, i: int) -> float
    { ret m_float::ldexp(n as m_float, i as c_int) as float }

/*
Function: ln1p

Returns the natural logarithm of `1+n` accurately,
even for very small values of `n`
*/
pure fn ln1p(n: float) -> float
    { ret m_float::ln1p(n as m_float) as float }

/*
Function: log10

Returns the logarithm to base 10 of `n`
*/
pure fn log10(n: float) -> float
    { ret m_float::log10(n as m_float) as float }

/*
Function: log2

Returns the logarithm to base 2 of `n`
*/
pure fn log2(n: float) -> float
    { ret m_float::log2(n as m_float) as float }

/*
Function: modf

Breaks `n` into integral and fractional parts such that both
have the same sign as `n`

The integral part is stored in `iptr`.

Returns:

The fractional part of `n`
*/
#[no(warn_trivial_casts)] // FIXME Implement
pure fn modf(n: float, &iptr: float) -> float { unsafe {
    ret m_float::modf(n as m_float, ptr::addr_of(iptr) as *m_float) as float
} }

/*
Function: frexp

Breaks `n` into a normalized fraction and an integral power of 2

The inegral part is stored in iptr.

The functions return a number x such that x has a magnitude in the interval
[1/2, 1) or 0, and `n == x*(2 to the power of exp)`.

Returns:

The fractional part of `n`
*/
pure fn frexp(n: float, &exp: c_int) -> float
    { ret m_float::frexp(n as m_float, exp) as float }

/*
Function: pow
*/
pure fn pow(v: float, e: float) -> float
    { ret m_float::pow(v as m_float, e as m_float) as float }


/*
Function: rint

Returns the integral value nearest to `x` (according to the
prevailing rounding mode) in floating-point format
*/
pure fn rint(x: float) -> float
    { ret m_float::rint(x as m_float) as float }

/*
Function: round


Return the integral value nearest to `x` rounding half-way
cases away from zero, regardless of the current rounding direction.
*/
pure fn round(x: float) -> float
    { ret m_float::round(x as m_float) as float }

/*
Function: sin

Returns the sine of an angle `x` (measured in rad)
*/
pure fn sin(x: float) -> float
    { ret m_float::sin(x as m_float) as float }

/*
Function: sinh

Returns the hyperbolic sine of an angle `x` (measured in rad)
*/
pure fn sinh(x: float) -> float
    { ret m_float::sinh(x as m_float) as float }

/*
Function: sqrt

Returns the square root of `x`
*/
pure fn sqrt(x: float) -> float
    { ret m_float::sqrt(x as m_float) as float }

/*
Function: tan

Returns the tangent of an angle `x` (measured in rad)

*/
pure fn tan(x: float) -> float
    { ret m_float::tan(x as m_float) as float }

/*
Function: tanh

Returns the hyperbolic tangent of an angle `x` (measured in rad)

*/
pure fn tanh(x: float) -> float
    { ret m_float::tanh(x as m_float) as float }

/*
Function: trunc

Returns the integral value nearest to but no larger in magnitude than `x`

*/
pure fn trunc(x: float) -> float
    { ret m_float::trunc(x as m_float) as float }

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

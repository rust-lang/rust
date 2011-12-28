/*
Module: f64

Floating point operations and constants for `f64`
*/

// PORT

import cmath::c_double::*;

type t = f64;


// These are not defined inside consts:: for consistency with
// the integer types

// PORT check per architecture

const radix: uint = 2u;

const mantissa_digits: uint = 53u;
const digits: uint = 15u;

const epsilon: f64 = 2.2204460492503131e-16_f64;

const min_value: f64 = 2.2250738585072014e-308_f64;
const max_value: f64 = 1.7976931348623157e+308_f64;

const min_exp: int = -1021;
const max_exp: int = 1024;

const min_10_exp: int = -307;
const max_10_exp: int = 308;

/* Const: NaN */
const NaN: f64 = 0.0_f64/0.0_f64;

/* Const: infinity */
const infinity: f64 = 1.0_f64/0.0_f64;

/* Const: neg_infinity */
const neg_infinity: f64 = -1.0_f64/0.0_f64;

/* Predicate: isNaN */
pure fn isNaN(f: f64) -> bool { f != f }

/* Function: add */
pure fn add(x: f64, y: f64) -> f64 { ret x + y; }

/* Function: sub */
pure fn sub(x: f64, y: f64) -> f64 { ret x - y; }

/* Function: mul */
pure fn mul(x: f64, y: f64) -> f64 { ret x * y; }

/* Function: div */
pure fn div(x: f64, y: f64) -> f64 { ret x / y; }

/* Function: rem */
pure fn rem(x: f64, y: f64) -> f64 { ret x % y; }

/* Predicate: lt */
pure fn lt(x: f64, y: f64) -> bool { ret x < y; }

/* Predicate: le */
pure fn le(x: f64, y: f64) -> bool { ret x <= y; }

/* Predicate: eq */
pure fn eq(x: f64, y: f64) -> bool { ret x == y; }

/* Predicate: ne */
pure fn ne(x: f64, y: f64) -> bool { ret x != y; }

/* Predicate: ge */
pure fn ge(x: f64, y: f64) -> bool { ret x >= y; }

/* Predicate: gt */
pure fn gt(x: f64, y: f64) -> bool { ret x > y; }

/*
Predicate: positive

Returns true if `x` is a positive number, including +0.0f640 and +Infinity.
 */
pure fn positive(x: f64) -> bool
    { ret x > 0.0f64 || (1.0f64/x) == infinity; }

/*
Predicate: negative

Returns true if `x` is a negative number, including -0.0f640 and -Infinity.
 */
pure fn negative(x: f64) -> bool
    { ret x < 0.0f64 || (1.0f64/x) == neg_infinity; }

/*
Predicate: nonpositive

Returns true if `x` is a negative number, including -0.0f640 and -Infinity.
(This is the same as `f64::negative`.)
*/
pure fn nonpositive(x: f64) -> bool {
  ret x < 0.0f64 || (1.0f64/x) == neg_infinity;
}

/*
Predicate: nonnegative

Returns true if `x` is a positive number, including +0.0f640 and +Infinity.
(This is the same as `f64::positive`.)
*/
pure fn nonnegative(x: f64) -> bool {
  ret x > 0.0f64 || (1.0f64/x) == infinity;
}

/* Module: consts */
mod consts {

    /*
    Const: pi

    Archimedes' constant
    */
    const pi: f64 = 3.14159265358979323846264338327950288_f64;

    /*
    Const: frac_pi_2

    pi/2.0
    */
    const frac_pi_2: f64 = 1.57079632679489661923132169163975144_f64;

    /*
    Const: frac_pi_4

    pi/4.0
    */
    const frac_pi_4: f64 = 0.785398163397448309615660845819875721_f64;

    /*
    Const: frac_1_pi

    1.0/pi
    */
    const frac_1_pi: f64 = 0.318309886183790671537767526745028724_f64;

    /*
    Const: frac_2_pi

    2.0/pi
    */
    const frac_2_pi: f64 = 0.636619772367581343075535053490057448_f64;

    /*
    Const: frac_2_sqrtpi

    2.0/sqrt(pi)
    */
    const frac_2_sqrtpi: f64 = 1.12837916709551257389615890312154517_f64;

    /*
    Const: sqrt2

    sqrt(2.0)
    */
    const sqrt2: f64 = 1.41421356237309504880168872420969808_f64;

    /*
    Const: frac_1_sqrt2

    1.0/sqrt(2.0)
    */
    const frac_1_sqrt2: f64 = 0.707106781186547524400844362104849039_f64;

    /*
    Const: e

    Euler's number
    */
    const e: f64 = 2.71828182845904523536028747135266250_f64;

    /*
    Const: log2_e

    log2(e)
    */
    const log2_e: f64 = 1.44269504088896340735992468100189214_f64;

    /*
    Const: log10_e

    log10(e)
    */
    const log10_e: f64 = 0.434294481903251827651128918916605082_f64;

    /*
    Const: ln_2

    ln(2.0)
    */
    const ln_2: f64 = 0.693147180559945309417232121458176568_f64;

    /*
    Const: ln_10

    ln(10.0)
    */
    const ln_10: f64 = 2.30258509299404568401799145468436421_f64;
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

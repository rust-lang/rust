#[doc = "Floating point operations and constants for `f64`"];

// PORT

import cmath::c_double::*;
import cmath::c_double_targ_consts::*;

// FIXME find out why these have to be exported explicitly

export add, sub, mul, div, rem, lt, le, gt, eq, eq, ne;
export is_positive, is_negative, is_nonpositive, is_nonnegative;
export is_zero, is_infinite, is_finite;
export NaN, is_NaN, infinity, neg_infinity;
export consts;
export logarithm;
export acos, asin, atan, atan2, cbrt, ceil, copysign, cos, cosh;
export erf, erfc, exp, expm1, exp2, abs, abs_sub;
export mul_add, fmax, fmin, nextafter, frexp, hypot, ldexp;
export lgamma, ln, log_radix, ln1p, log10, log2, ilog_radix;
export modf, pow, round, sin, sinh, sqrt, tan, tanh, tgamma, trunc;
export signbit;
export epsilon;

type t = f64;

// These are not defined inside consts:: for consistency with
// the integer types

// PORT check per architecture

// FIXME obtain these in a different way

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

const NaN: f64 = 0.0_f64/0.0_f64;

const infinity: f64 = 1.0_f64/0.0_f64;

const neg_infinity: f64 = -1.0_f64/0.0_f64;

pure fn is_NaN(f: f64) -> bool { f != f }

pure fn add(x: f64, y: f64) -> f64 { ret x + y; }

pure fn sub(x: f64, y: f64) -> f64 { ret x - y; }

pure fn mul(x: f64, y: f64) -> f64 { ret x * y; }

pure fn div(x: f64, y: f64) -> f64 { ret x / y; }

pure fn rem(x: f64, y: f64) -> f64 { ret x % y; }

pure fn lt(x: f64, y: f64) -> bool { ret x < y; }

pure fn le(x: f64, y: f64) -> bool { ret x <= y; }

pure fn eq(x: f64, y: f64) -> bool { ret x == y; }

pure fn ne(x: f64, y: f64) -> bool { ret x != y; }

pure fn ge(x: f64, y: f64) -> bool { ret x >= y; }

pure fn gt(x: f64, y: f64) -> bool { ret x > y; }

#[doc(
  brief = "Returns true if `x` is a positive number, including \
           +0.0f640 and +Infinity."
)]
pure fn is_positive(x: f64) -> bool
    { ret x > 0.0f64 || (1.0f64/x) == infinity; }

#[doc(
  brief = "Returns true if `x` is a negative number, including \
           -0.0f640 and -Infinity."
)]
pure fn is_negative(x: f64) -> bool
    { ret x < 0.0f64 || (1.0f64/x) == neg_infinity; }

#[doc(
  brief = "Returns true if `x` is a negative number, including \
           -0.0f640 and -Infinity. (This is the same as \
           `f64::negative`.)"
)]
pure fn is_nonpositive(x: f64) -> bool {
  ret x < 0.0f64 || (1.0f64/x) == neg_infinity;
}

#[doc(
  brief = "Returns true if `x` is a positive number, including \
           +0.0f640 and +Infinity.(This is the same as \
           `f64::positive`.)"
)]
pure fn is_nonnegative(x: f64) -> bool {
  ret x > 0.0f64 || (1.0f64/x) == infinity;
}

#[doc(
  brief = "Returns true if `x` is a zero number (positive or \
           negative zero)"
)]
pure fn is_zero(x: f64) -> bool {
    ret x == 0.0f64 || x == -0.0f64;
}

#[doc(
  brief = "Returns true if `x`is an infinite number."
)]
pure fn is_infinite(x: f64) -> bool {
    ret x == infinity || x == neg_infinity;
}

#[doc(
  brief = "Returns true if `x`is a finite number."
)]
pure fn is_finite(x: f64) -> bool {
    ret !(is_NaN(x) || is_infinite(x));
}

// FIXME add is_normal, is_subnormal, and fpclassify

/* Module: consts */
mod consts {

    // FIXME replace with mathematical constants from cmath

    #[doc(
      brief = "Archimedes' constant"
    )]
    const pi: f64 = 3.14159265358979323846264338327950288_f64;

    #[doc(
      brief = "pi/2.0"
    )]
    const frac_pi_2: f64 = 1.57079632679489661923132169163975144_f64;

    #[doc(
      brief = "pi/4.0"
    )]
    const frac_pi_4: f64 = 0.785398163397448309615660845819875721_f64;

    #[doc(
      brief = "1.0/pi"
    )]
    const frac_1_pi: f64 = 0.318309886183790671537767526745028724_f64;

    #[doc(
      brief = "2.0/pi"
    )]
    const frac_2_pi: f64 = 0.636619772367581343075535053490057448_f64;

    #[doc(
      brief = "2.0/sqrt(pi)"
    )]
    const frac_2_sqrtpi: f64 = 1.12837916709551257389615890312154517_f64;

    #[doc(
      brief = "sqrt(2.0)"
    )]
    const sqrt2: f64 = 1.41421356237309504880168872420969808_f64;

    #[doc(
      brief = "1.0/sqrt(2.0)"
    )]
    const frac_1_sqrt2: f64 = 0.707106781186547524400844362104849039_f64;

    #[doc(
      brief = "Euler's number"
    )]
    const e: f64 = 2.71828182845904523536028747135266250_f64;

    #[doc(
      brief = "log2(e)"
    )]
    const log2_e: f64 = 1.44269504088896340735992468100189214_f64;

    #[doc(
      brief = "log10(e)"
    )]
    const log10_e: f64 = 0.434294481903251827651128918916605082_f64;

    #[doc(
      brief = "ln(2.0)"
    )]
    const ln_2: f64 = 0.693147180559945309417232121458176568_f64;

    #[doc(
      brief = "ln(10.0)"
    )]
    const ln_10: f64 = 2.30258509299404568401799145468436421_f64;
}

pure fn signbit(x: f64) -> int {
    if is_negative(x) { ret 1; } else { ret 0; }
}

#[cfg(target_os="linux")]
#[cfg(target_os="macos")]
#[cfg(target_os="win32")]
pure fn logarithm(n: f64, b: f64) -> f64 {
    // FIXME check if it is good to use log2 instead of ln here;
    // in theory should be faster since the radix is 2
    ret log2(n) / log2(b);
}

#[cfg(target_os="freebsd")]
pure fn logarithm(n: f64, b: f64) -> f64 {
    ret ln(n) / ln(b);
}

#[cfg(target_os="freebsd")]
pure fn log2(n: f64) -> f64 {
    ret ln(n) / consts::ln_2;
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

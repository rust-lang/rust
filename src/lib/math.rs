/* Module: math */

export consts;
export min, max;

export f32, f64;

// Currently this module supports from -lmath:
// C95 + log2 + log1p + trunc + round + rint

export
    acos, asin, atan, atan2, ceil, cos, cosh, exp, abs, floor, fmod, frexp,
    ldexp, ln, ln1p, log10, log2, modf, rint, round, pow, sin, sinh, sqrt,
    tan, tanh, trunc;

// These two must match in width according to architecture

import ctypes::c_float;
import ctypes::c_int;
import c_float = f64;


#[link_name = "m"]
#[abi = "cdecl"]
native mod f64 {

    // Alpabetically sorted by link_name

    pure fn acos(n: f64) -> f64;
    pure fn asin(n: f64) -> f64;
    pure fn atan(n: f64) -> f64;
    pure fn atan2(a: f64, b: f64) -> f64;
    pure fn ceil(n: f64) -> f64;
    pure fn cos(n: f64) -> f64;
    pure fn cosh(n: f64) -> f64;
    pure fn exp(n: f64) -> f64;
    #[link_name="fabs"] pure fn abs(n: f64) -> f64;
    pure fn floor(n: f64) -> f64;
    pure fn fmod(x: f64, y: f64) -> f64;
    pure fn frexp(n: f64, &value: c_int) -> f64;
    pure fn ldexp(x: f64, n: c_int) -> f64;
    #[link_name="log"] pure fn ln(n: f64) -> f64;
    #[link_name="log1p"] pure fn ln1p(n: f64) -> f64;
    pure fn log10(n: f64) -> f64;
    pure fn log2(n: f64) -> f64;
    pure fn modf(n: f64, &iptr: f64) -> f64;
    pure fn pow(n: f64, e: f64) -> f64;
    pure fn rint(n: f64) -> f64;
    pure fn round(n: f64) -> f64;
    pure fn sin(n: f64) -> f64;
    pure fn sinh(n: f64) -> f64;
    pure fn sqrt(n: f64) -> f64;
    pure fn tan(n: f64) -> f64;
    pure fn tanh(n: f64) -> f64;
    pure fn trunc(n: f64) -> f64;
}

#[link_name = "m"]
#[abi = "cdecl"]
native mod f32 {

    // Alpabetically sorted by link_name

    #[link_name="acosf"] pure fn acos(n: f32) -> f32;
    #[link_name="asinf"] pure fn asin(n: f32) -> f32;
    #[link_name="atanf"] pure fn atan(n: f32) -> f32;
    #[link_name="atan2f"] pure fn atan2(a: f32, b: f32) -> f32;
    #[link_name="ceilf"] pure fn ceil(n: f32) -> f32;
    #[link_name="cosf"] pure fn cos(n: f32) -> f32;
    #[link_name="coshf"] pure fn cosh(n: f32) -> f32;
    #[link_name="expf"] pure fn exp(n: f32) -> f32;
    #[link_name="fabsf"] pure fn abs(n: f32) -> f32;
    #[link_name="floorf"] pure fn floor(n: f32) -> f32;
    #[link_name="frexpf"] pure fn frexp(n: f64, &value: c_int) -> f32;
    #[link_name="fmodf"] pure fn fmod(x: f32, y: f32) -> f32;
    #[link_name="ldexpf"] pure fn ldexp(x: f32, n: c_int) -> f32;
    #[link_name="logf"] pure fn ln(n: f32) -> f32;
    #[link_name="log1p"] pure fn ln1p(n: f64) -> f64;
    #[link_name="log2f"] pure fn log2(n: f32) -> f32;
    #[link_name="log10f"] pure fn log10(n: f32) -> f32;
    #[link_name="modff"] pure fn modf(n: f32, &iptr: f32) -> f32;
    #[link_name="powf"] pure fn pow(n: f32, e: f32) -> f32;
    #[link_name="rintf"] pure fn rint(n: f32) -> f32;
    #[link_name="roundf"] pure fn round(n: f32) -> f32;
    #[link_name="sinf"] pure fn sin(n: f32) -> f32;
    #[link_name="sinhf"] pure fn sinh(n: f32) -> f32;
    #[link_name="sqrtf"] pure fn sqrt(n: f32) -> f32;
    #[link_name="tanf"] pure fn tan(n: f32) -> f32;
    #[link_name="tanhf"] pure fn tanh(n: f32) -> f32;
    #[link_name="truncf"] pure fn trunc(n: f32) -> f32;
}

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
    { c_float::acos(x as c_float) as float }

/*
Function: asin

Returns the arcsine of an angle (measured in rad)
*/
pure fn asin(x: float) -> float
    { c_float::asin(x as c_float) as float }

/*
Function: atan

Returns the arctangents of an angle (measured in rad)
*/
pure fn atan(x: float) -> float
    { c_float::atan(x as c_float) as float }


/*
Function: atan2

Returns the arctangent of an angle (measured in rad)
*/
pure fn atan2(y: float, x: float) -> float
    { c_float::atan2(y as c_float, x as c_float) as float }

/*
Function: ceil

Returns the smallest integral value less than or equal to `n`
*/
pure fn ceil(n: float) -> float
    { c_float::ceil(n as c_float) as float }

/*
Function: cos

Returns the cosine of an angle `x` (measured in rad)
*/
pure fn cos(x: float) -> float
    { c_float::cos(x as c_float) as float }

/*
Function: cosh

Returns the hyperbolic cosine of `x`

*/
pure fn cosh(x: float) -> float
    { c_float::cosh(x as c_float) as float }


/*
Function: exp

Returns `consts::e` to the power of `n*
*/
pure fn exp(n: float) -> float
    { c_float::exp(n as c_float) as float }

/*
Function: abs

Returns the absolute value of  `n`
*/
pure fn abs(n: float) -> float
    { c_float::abs(n as c_float) as float }

/*
Function: floor

Returns the largest integral value less than or equal to `n`
*/
pure fn floor(n: float) -> float
    { c_float::floor(n as c_float) as float }

/*
Function: fmod

Returns the floating-point remainder of `x/y`
*/
pure fn fmod(x: float, y: float) -> float
    { c_float::fmod(x as c_float, y as c_float) as float }

/*
Function: ln

Returns the natural logaritm of `n`
*/
pure fn ln(n: float) -> float
    { c_float::ln(n as c_float) as float }

/*
Function: ldexp

Returns `x` multiplied by 2 to the power of `n`
*/
pure fn ldexp(n: float, i: int) -> float
    { c_float::ldexp(n as c_float, i as c_int) as float }

/*
Function: ln1p

Returns the natural logarithm of `1+n` accurately,
even for very small values of `n`
*/
pure fn ln1p(n: float) -> float
    { c_float::ln1p(n as c_float) as float }

/*
Function: log10

Returns the logarithm to base 10 of `n`
*/
pure fn log10(n: float) -> float
    { c_float::log10(n as c_float) as float }

/*
Function: log2

Returns the logarithm to base 2 of `n`
*/
pure fn log2(n: float) -> float
    { c_float::log2(n as c_float) as float }


/*
Function: modf

Breaks `n` into integral and fractional parts such that both
have the same sign as `n`

The integral part is stored in `iptr`.

Returns:

The fractional part of `n`
*/
pure fn modf(n: float, &iptr: float) -> float {
    unchecked {
        let f = iptr as c_float;
        let r = c_float::modf(n as c_float, f) as float;
        iptr  = f as float;
        ret r;
    }
}

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
    { c_float::frexp(n as c_float, exp) as float }

/*
Function: pow
*/
pure fn pow(v: float, e: float) -> float
    { c_float::pow(v as c_float, e as c_float) as float }


/*
Function: rint

Returns the integral value nearest to `x` (according to the
prevailing rounding mode) in floating-point format
*/
pure fn rint(x: float) -> float
    { c_float::rint(x as c_float) as float }

/*
Function: round


Return the integral value nearest to `x` rounding half-way
cases away from zero, regardless of the current rounding direction.
*/
pure fn round(x: float) -> float
    { c_float::round(x as c_float) as float }

/*
Function: sin

Returns the sine of an angle `x` (measured in rad)
*/
pure fn sin(x: float) -> float
    { c_float::sin(x as c_float) as float }

/*
Function: sinh

Returns the hyperbolic sine of an angle `x` (measured in rad)
*/
pure fn sinh(x: float) -> float
    { c_float::sinh(x as c_float) as float }

/*
Function: sqrt

Returns the square root of `x`
*/
pure fn sqrt(x: float) -> float
    { c_float::sqrt(x as c_float) as float }

/*
Function: tan

Returns the tangent of an angle `x` (measured in rad)

*/
pure fn tan(x: float) -> float
    { c_float::tan(x as c_float) as float }

/*
Function: tanh

Returns the hyperbolic tangent of an angle `x` (measured in rad)

*/
pure fn tanh(x: float) -> float
    { c_float::tanh(x as c_float) as float }

/*
Function: trunc

Returns the integral value nearest to but no larger in magnitude than `x`

*/
pure fn trunc(x: float) -> float
    { c_float::trunc(x as c_float) as float }





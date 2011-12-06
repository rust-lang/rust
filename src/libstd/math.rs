/*

Module: math

Floating point operations and constants for `float`s
*/

export consts;
export min, max;

// Currently this module supports from -lmath:
// C95 + log2 + log1p + trunc + round + rint

export
    acos, asin, atan, atan2, ceil, cos, cosh, exp, abs, floor, fmod, frexp,
    ldexp, ln, ln1p, log10, log2, modf, rint, round, pow, sin, sinh, sqrt,
    tan, tanh, trunc;

// These two must match in width according to architecture

import ctypes::m_float;
import ctypes::c_int;
import m_float = math_f64;

// FIXME replace with redirect to m_float::consts::FOO as soon as it works
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
    { m_float::acos(x as m_float) as float }

/*
Function: asin

Returns the arcsine of an angle (measured in rad)
*/
pure fn asin(x: float) -> float
    { m_float::asin(x as m_float) as float }

/*
Function: atan

Returns the arctangents of an angle (measured in rad)
*/
pure fn atan(x: float) -> float
    { m_float::atan(x as m_float) as float }


/*
Function: atan2

Returns the arctangent of an angle (measured in rad)
*/
pure fn atan2(y: float, x: float) -> float
    { m_float::atan2(y as m_float, x as m_float) as float }

/*
Function: ceil

Returns the smallest integral value less than or equal to `n`
*/
pure fn ceil(n: float) -> float
    { m_float::ceil(n as m_float) as float }

/*
Function: cos

Returns the cosine of an angle `x` (measured in rad)
*/
pure fn cos(x: float) -> float
    { m_float::cos(x as m_float) as float }

/*
Function: cosh

Returns the hyperbolic cosine of `x`

*/
pure fn cosh(x: float) -> float
    { m_float::cosh(x as m_float) as float }


/*
Function: exp

Returns `consts::e` to the power of `n*
*/
pure fn exp(n: float) -> float
    { m_float::exp(n as m_float) as float }

/*
Function: abs

Returns the absolute value of  `n`
*/
pure fn abs(n: float) -> float
    { m_float::abs(n as m_float) as float }

/*
Function: floor

Returns the largest integral value less than or equal to `n`
*/
pure fn floor(n: float) -> float
    { m_float::floor(n as m_float) as float }

/*
Function: fmod

Returns the floating-point remainder of `x/y`
*/
pure fn fmod(x: float, y: float) -> float
    { m_float::fmod(x as m_float, y as m_float) as float }

/*
Function: ln

Returns the natural logaritm of `n`
*/
pure fn ln(n: float) -> float
    { m_float::ln(n as m_float) as float }

/*
Function: ldexp

Returns `x` multiplied by 2 to the power of `n`
*/
pure fn ldexp(n: float, i: int) -> float
    { m_float::ldexp(n as m_float, i as c_int) as float }

/*
Function: ln1p

Returns the natural logarithm of `1+n` accurately,
even for very small values of `n`
*/
pure fn ln1p(n: float) -> float
    { m_float::ln1p(n as m_float) as float }

/*
Function: log10

Returns the logarithm to base 10 of `n`
*/
pure fn log10(n: float) -> float
    { m_float::log10(n as m_float) as float }

/*
Function: log2

Returns the logarithm to base 2 of `n`
*/
pure fn log2(n: float) -> float
    { m_float::log2(n as m_float) as float }


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
        let f = iptr as m_float;
        let r = m_float::modf(n as m_float, f) as float;
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
    { m_float::frexp(n as m_float, exp) as float }

/*
Function: pow
*/
pure fn pow(v: float, e: float) -> float
    { m_float::pow(v as m_float, e as m_float) as float }


/*
Function: rint

Returns the integral value nearest to `x` (according to the
prevailing rounding mode) in floating-point format
*/
pure fn rint(x: float) -> float
    { m_float::rint(x as m_float) as float }

/*
Function: round


Return the integral value nearest to `x` rounding half-way
cases away from zero, regardless of the current rounding direction.
*/
pure fn round(x: float) -> float
    { m_float::round(x as m_float) as float }

/*
Function: sin

Returns the sine of an angle `x` (measured in rad)
*/
pure fn sin(x: float) -> float
    { m_float::sin(x as m_float) as float }

/*
Function: sinh

Returns the hyperbolic sine of an angle `x` (measured in rad)
*/
pure fn sinh(x: float) -> float
    { m_float::sinh(x as m_float) as float }

/*
Function: sqrt

Returns the square root of `x`
*/
pure fn sqrt(x: float) -> float
    { m_float::sqrt(x as m_float) as float }

/*
Function: tan

Returns the tangent of an angle `x` (measured in rad)

*/
pure fn tan(x: float) -> float
    { m_float::tan(x as m_float) as float }

/*
Function: tanh

Returns the hyperbolic tangent of an angle `x` (measured in rad)

*/
pure fn tanh(x: float) -> float
    { m_float::tanh(x as m_float) as float }

/*
Function: trunc

Returns the integral value nearest to but no larger in magnitude than `x`

*/
pure fn trunc(x: float) -> float
    { m_float::trunc(x as m_float) as float }





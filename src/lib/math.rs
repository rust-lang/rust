/* Module: math */

export consts;
export min, max;

export f32, f64;

// Currently this module supports from -lmath
// C95 - frexp - ldexp - fmod - modf + log2 + log1p

export
    acos, asin, atan, atan2, ceil, cos, cosh, exp, abs, floor,
    ln, ln1p, log10, log2, pow, sin, sinh, sqrt, tan, tanh;

// These two must match in width according to architecture

import ctypes::c_float;
import c_float = f64;


#[link_name = "m"]
#[abi = "cdecl"]
native mod f64 {

    // Alpabetically sorted by link_name

    fn acos(n: f64) -> f64;
    fn asin(n: f64) -> f64;
    fn atan(n: f64) -> f64;
    fn atan2(a: f64, b: f64) -> f64;
    fn ceil(n: f64) -> f64;
    fn cos(n: f64) -> f64;
    fn cosh(n: f64) -> f64;
    fn exp(n: f64) -> f64;
    #[link_name="fabs"] fn abs(n: f64) -> f64; 
    fn floor(n: f64) -> f64;
    #[link_name="log"] fn ln(n: f64) -> f64;
    #[link_name="log1p"] fn ln1p(n: f64) -> f64;
    fn log10(n: f64) -> f64;
    fn log2(n: f64) -> f64;
    fn pow(n: f64, e: f64) -> f64;
    fn sin(n: f64) -> f64;
    fn sinh(n: f64) -> f64;
    fn sqrt(n: f64) -> f64;
    fn tan(n: f64) -> f64;
    fn tanh(n: f64) -> f64;
}

#[link_name = "m"]
#[abi = "cdecl"]
native mod f32 {

    // Alpabetically sorted by link_name

    #[link_name="acosf"] fn acos(n: f32) -> f32;
    #[link_name="asinf"] fn asin(n: f32) -> f32;
    #[link_name="atanf"] fn atan(n: f32) -> f32;
    #[link_name="atan2f"] fn atan2(a: f32, b: f32) -> f32;
    #[link_name="ceilf"] fn ceil(n: f32) -> f32;
    #[link_name="cosf"] fn cos(n: f32) -> f32;
    #[link_name="coshf"] fn cosh(n: f32) -> f32;
    #[link_name="expf"] fn exp(n: f32) -> f32;
    #[link_name="fabsf"] fn abs(n: f32) -> f32;
    #[link_name="floorf"] fn floor(n: f32) -> f32;
    #[link_name="powf"] fn pow(n: f32, e: f32) -> f32;
    #[link_name="sinf"] fn sin(n: f32) -> f32;
    #[link_name="sinhf"] fn sinh(n: f32) -> f32;
    #[link_name="sqrtf"] fn sqrt(n: f32) -> f32;
    #[link_name="tanf"] fn tan(n: f32) -> f32;
    #[link_name="tanhf"] fn tanh(n: f32) -> f32;
    #[link_name="logf"] fn ln(n: f32) -> f32;
    #[link_name="log1p"] fn ln1p(n: f64) -> f64;
    #[link_name="log2f"] fn log2(n: f32) -> f32;
    #[link_name="log10f"] fn log10(n: f32) -> f32;
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
    { unsafe { c_float::acos(x as c_float) as float } }

/*
Function: asin

Returns the arcsine of an angle (measured in rad)
*/
pure fn asin(x: float) -> float
    { unsafe { c_float::asin(x as c_float) as float } }

/*
Function: atan

Returns the arctangents of an angle (measured in rad)
*/
pure fn atan(x: float) -> float
    { unsafe { c_float::atan(x as c_float) as float } }


/*
Function: atan2

Returns the arctangent of an angle (measured in rad)
*/
pure fn atan2(y: float, x: float) -> float
    { unsafe { c_float::atan2(y as c_float, x as c_float) as float } }

/*
Function: ceil

Returns:

The smallest integral value less than or equal to `n`
*/
pure fn ceil(n: float) -> float
    { unsafe { c_float::ceil(n as c_float) as float } }

/*
Function: cos

Returns the cosine of an angle `x` (measured in rad)
*/
pure fn cos(x: float) -> float
    { unsafe { c_float::cos(x as c_float) as float } }

/*
Function: cosh

Returns the hyperbolic cosine of `x`

*/
pure fn cosh(x: float) -> float
    { unsafe { c_float::cosh(x as c_float) as float } }


/*
Function: exp

Returns:

e to the power of `n*
*/
pure fn exp(n: float) -> float
    { unsafe { c_float::exp(n as c_float) as float } }

/*
Function: abs

Returns:

The absolute value of  `n`

*/
pure fn abs(n: float) -> float
    { unsafe { c_float::abs(n as c_float) as float } }

/*
Function: floor

Returns:

The largest integral value less than or equal to `n`
*/
pure fn floor(n: float) -> float
    { unsafe { c_float::floor(n as c_float) as float } }

/*
Function: ln

Returns the natural logaritm of `n`
*/
pure fn ln(n: float) -> float
    { unsafe { c_float::ln(n as c_float) as float } }

/*
Function: ln1p

Returns the natural logarithm of `1+n` accurately,
even for very small values of `n`
*/
pure fn ln1p(n: float) -> float
    { unsafe { c_float::ln1p(n as c_float) as float } }

/*
Function: log10

Returns the logarithm to base 10 of `n`
*/
pure fn log10(n: float) -> float
    { unsafe { c_float::log10(n as c_float) as float } }

/*
Function: log2

Returns the logarithm to base 2 of `n`
*/
pure fn log2(n: float) -> float
    { unsafe { c_float::log2(n as c_float) as float } }

/*
Function: pow
*/
pure fn pow(v: float, e: float) -> float
    { unsafe { c_float::pow(v as c_float, e as c_float) as float } }


/*
Function: sin

Returns the sine of an angle `x` (measured in rad)
*/
pure fn sin(x: float) -> float
    { unsafe { c_float::sin(x as c_float) as float } }

/*
Function: sinh

Returns the hyperbolic sine of an angle `x` (measured in rad)
*/
pure fn sinh(x: float) -> float
    { unsafe { c_float::sinh(x as c_float) as float } }

/*
Function: sqrt

Returns the square root of `x`
*/
pure fn sqrt(x: float) -> float
    { unsafe { c_float::sqrt(x as c_float) as float } }

/*
Function: tan

Returns the tangent of an angle `x` (measured in rad)

*/
pure fn tan(x: float) -> float
    { unsafe { c_float::tan(x as c_float) as float } }

/*
Function: tanh

Returns the hyperbolic tangent of an angle `x` (measured in rad)

*/
pure fn tanh(x: float) -> float
    { unsafe { c_float::tanh(x as c_float) as float } }






/*
Module: math_f64

Floating point operations and constants for `f64`s

This exposes the same operations as `math`, just for `f64` even though
they do not show up in the docs right now!
*/

import cmath::f64::*;

export
    acos, asin, atan, atan2, ceil, cos, cosh, exp, abs, floor, fmod,
    frexp, ldexp, ln, ln1p, log10, log2, modf, rint, round, pow, sin,
    sinh, sqrt, tan, tanh, trunc;

export consts;

/* Module: consts */
mod consts {

    /*
    Const: pi

    Archimedes' constant
    */
    const pi: f64 = 3.14159265358979323846264338327950288f64;

    /*
    Const: frac_pi_2

    pi/2.0
    */
    const frac_pi_2: f64 = 1.57079632679489661923132169163975144f64;

    /*
    Const: frac_pi_4

    pi/4.0
    */
    const frac_pi_4: f64 = 0.785398163397448309615660845819875721f64;

    /*
    Const: frac_1_pi

    1.0/pi
    */
    const frac_1_pi: f64 = 0.318309886183790671537767526745028724f64;

    /*
    Const: frac_2_pi

    2.0/pi
    */
    const frac_2_pi: f64 = 0.636619772367581343075535053490057448f64;

    /*
    Const: frac_2_sqrtpi

    2.0/sqrt(pi)
    */
    const frac_2_sqrtpi: f64 = 1.12837916709551257389615890312154517f64;

    /*
    Const: sqrt2

    sqrt(2.0)
    */
    const sqrt2: f64 = 1.41421356237309504880168872420969808f64;

    /*
    Const: frac_1_sqrt2

    1.0/sqrt(2.0)
    */
    const frac_1_sqrt2: f64 = 0.707106781186547524400844362104849039f64;

    /*
    Const: e

    Euler's number
    */
    const e: f64 = 2.71828182845904523536028747135266250f64;

    /*
    Const: log2_e

    log2(e)
    */
    const log2_e: f64 = 1.44269504088896340735992468100189214f64;

    /*
    Const: log10_e

    log10(e)
    */
    const log10_e: f64 = 0.434294481903251827651128918916605082f64;

    /*
    Const: ln_2

    ln(2.0)
    */
    const ln_2: f64 = 0.693147180559945309417232121458176568f64;

    /*
    Const: ln_10

    ln(10.0)
    */
    const ln_10: f64 = 2.30258509299404568401799145468436421f64;
}
export c_float;
export c_double;

// FIXME export c_float_math_consts;
// FIXME export c_double_math_consts;

export c_float_targ_consts;
export c_double_targ_consts;

import ctypes::c_int;
import ctypes::c_float;
import ctypes::c_double;

// function names are almost identical to C's libmath, a few have been
// renamed, grep for "rename:"

#[link_name = "m"]
#[abi = "cdecl"]
native mod c_double {

    // Alpabetically sorted by link_name

    pure fn acos(n: c_double) -> c_double;
    pure fn asin(n: c_double) -> c_double;
    pure fn atan(n: c_double) -> c_double;
    pure fn atan2(a: c_double, b: c_double) -> c_double;
    pure fn cbrt(n: c_double) -> c_double;
    pure fn ceil(n: c_double) -> c_double;
    pure fn copysign(x: c_double, y: c_double) -> c_double;
    pure fn cos(n: c_double) -> c_double;
    pure fn cosh(n: c_double) -> c_double;
    pure fn erf(n: c_double) -> c_double;
    pure fn erfc(n: c_double) -> c_double;
    pure fn exp(n: c_double) -> c_double;
    pure fn expm1(n: c_double) -> c_double;
    pure fn exp2(n: c_double) -> c_double;
    #[link_name="fabs"] pure fn abs(n: c_double) -> c_double;
    // rename: for clarity and consistency with add/sub/mul/div
    #[link_name="fdim"] pure fn abs_sub(a: c_double, b: c_double) -> c_double;
    pure fn floor(n: c_double) -> c_double;
    // rename: for clarity and consistency with add/sub/mul/div
    #[link_name="fma"] pure fn mul_add(a: c_double, b: c_double,
                                       c: c_double) -> c_double;
    #[link_name="fmax"] pure fn fmax(a: c_double, b: c_double) -> c_double;
    #[link_name="fmin"] pure fn fmin(a: c_double, b: c_double) -> c_double;
    pure fn nextafter(x: c_double, y: c_double) -> c_double;
    pure fn frexp(n: c_double, &value: c_int) -> c_double;
    pure fn hypot(x: c_double, y: c_double) -> c_double;
    pure fn ldexp(x: c_double, n: c_int) -> c_double;
    #[link_name="lgamma_r"] pure fn lgamma(n: c_double,
                                           &sign: c_int) -> c_double;
    // renamed: log is a reserved keyword; ln seems more natural, too
    #[link_name="log"] pure fn ln(n: c_double) -> c_double;
    // renamed: "logb" /often/ is confused for log2 by beginners
    #[link_name="logb"] pure fn log_radix(n: c_double) -> c_double;
    // renamed: to be consitent with log as ln
    #[link_name="log1p"] pure fn ln1p(n: c_double) -> c_double;
    pure fn log10(n: c_double) -> c_double;
    #[cfg(target_os="linux")]
    #[cfg(target_os="macos")]
    #[cfg(target_os="win32")]
    pure fn log2(n: c_double) -> c_double;
    #[link_name="ilogb"] pure fn ilog_radix(n: c_double) -> c_int;
    pure fn modf(n: c_double, &iptr: c_double) -> c_double;
    pure fn pow(n: c_double, e: c_double) -> c_double;
// FIXME enable when rounding modes become available
//    pure fn rint(n: c_double) -> c_double;
    pure fn round(n: c_double) -> c_double;
    // rename: for consistency with logradix
    #[link_name="scalbn"] pure fn ldexp_radix(n: c_double, i: c_int) ->
        c_double;
    pure fn sin(n: c_double) -> c_double;
    pure fn sinh(n: c_double) -> c_double;
    pure fn sqrt(n: c_double) -> c_double;
    pure fn tan(n: c_double) -> c_double;
    pure fn tanh(n: c_double) -> c_double;
    pure fn tgamma(n: c_double) -> c_double;
    pure fn trunc(n: c_double) -> c_double;

    // These are commonly only available for doubles

    pure fn j0(n: c_double) -> c_double;
    pure fn j1(n: c_double) -> c_double;
    pure fn jn(i: c_int, n: c_double) -> c_double;

    pure fn y0(n: c_double) -> c_double;
    pure fn y1(n: c_double) -> c_double;
    pure fn yn(i: c_int, n: c_double) -> c_double;
}

#[link_name = "m"]
#[abi = "cdecl"]
native mod c_float {

    // Alpabetically sorted by link_name

    #[link_name="acosf"] pure fn acos(n: c_float) -> c_float;
    #[link_name="asinf"] pure fn asin(n: c_float) -> c_float;
    #[link_name="atanf"] pure fn atan(n: c_float) -> c_float;
    #[link_name="atan2f"] pure fn atan2(a: c_float, b: c_float) -> c_float;
    #[link_name="cbrtf"] pure fn cbrt(n: c_float) -> c_float;
    #[link_name="ceilf"] pure fn ceil(n: c_float) -> c_float;
    #[link_name="copysignf"] pure fn copysign(x: c_float,
                                              y: c_float) -> c_float;
    #[link_name="cosf"] pure fn cos(n: c_float) -> c_float;
    #[link_name="coshf"] pure fn cosh(n: c_float) -> c_float;
    #[link_name="erff"] pure fn erf(n: c_float) -> c_float;
    #[link_name="erfcf"] pure fn erfc(n: c_float) -> c_float;
    #[link_name="expf"] pure fn exp(n: c_float) -> c_float;
    #[link_name="expm1f"]pure fn expm1(n: c_float) -> c_float;
    #[link_name="exp2f"] pure fn exp2(n: c_float) -> c_float;
    #[link_name="fabsf"] pure fn abs(n: c_float) -> c_float;
    #[link_name="fdimf"] pure fn abs_sub(a: c_float, b: c_float) -> c_float;
    #[link_name="floorf"] pure fn floor(n: c_float) -> c_float;
    #[link_name="frexpf"] pure fn frexp(n: c_float,
                                        &value: c_int) -> c_float;
    #[link_name="fmaf"] pure fn mul_add(a: c_float,
                                        b: c_float, c: c_float) -> c_float;
    #[link_name="fmaxf"] pure fn fmax(a: c_float, b: c_float) -> c_float;
    #[link_name="fminf"] pure fn fmin(a: c_float, b: c_float) -> c_float;
    #[link_name="nextafterf"] pure fn nextafter(x: c_float,
                                                y: c_float) -> c_float;
    #[link_name="hypotf"] pure fn hypot(x: c_float, y: c_float) -> c_float;
    #[link_name="ldexpf"] pure fn ldexp(x: c_float, n: c_int) -> c_float;

    #[cfg(target_os="linux")]
    #[cfg(target_os="macos")]
    #[link_name="lgammaf_r"] pure fn lgamma(n: c_float,
                                            &sign: c_int) -> c_float;

    #[cfg(target_os="win32")]
    #[link_name="__lgammaf_r"] pure fn lgamma(n: c_float,
                                              &sign: c_int) -> c_float;

    #[link_name="logf"] pure fn ln(n: c_float) -> c_float;
    #[link_name="logbf"] pure fn log_radix(n: c_float) -> c_float;
    #[link_name="log1pf"] pure fn ln1p(n: c_float) -> c_float;
    #[cfg(target_os="linux")]
    #[cfg(target_os="macos")]
    #[cfg(target_os="win32")]
    #[link_name="log2f"] pure fn log2(n: c_float) -> c_float;
    #[link_name="log10f"] pure fn log10(n: c_float) -> c_float;
    #[link_name="ilogbf"] pure fn ilog_radix(n: c_float) -> c_int;
    #[link_name="modff"] pure fn modf(n: c_float,
                                      &iptr: c_float) -> c_float;
    #[link_name="powf"] pure fn pow(n: c_float, e: c_float) -> c_float;
// FIXME enable when rounding modes become available
//    #[link_name="rintf"] pure fn rint(n: c_float) -> c_float;
    #[link_name="roundf"] pure fn round(n: c_float) -> c_float;
    #[link_name="scalbnf"] pure fn ldexp_radix(n: c_float, i: c_int)
        -> c_float;
    #[link_name="sinf"] pure fn sin(n: c_float) -> c_float;
    #[link_name="sinhf"] pure fn sinh(n: c_float) -> c_float;
    #[link_name="sqrtf"] pure fn sqrt(n: c_float) -> c_float;
    #[link_name="tanf"] pure fn tan(n: c_float) -> c_float;
    #[link_name="tanhf"] pure fn tanh(n: c_float) -> c_float;
    #[link_name="tgammaf"] pure fn tgamma(n: c_float) -> c_float;
    #[link_name="truncf"] pure fn trunc(n: c_float) -> c_float;
}

// PORT check these by running src/etc/machconsts.c for your architecture

// FIXME obtain machine float/math constants automatically

mod c_float_targ_consts {
    const radix: uint = 2u;
    const mantissa_digits: uint = 24u;
    const digits: uint = 6u;
    const min_exp: uint = -125u;
    const max_exp: uint = 128u;
    const min_10_exp: int = -37;
    const max_10_exp: int = 38;
    // FIXME this is wrong! replace with hexadecimal (%a) constants below
    const min_value: f32 = 1.175494e-38_f32;
    const max_value: f32 = 3.402823e+38_f32;
    const epsilon: f32 = 0.000000_f32;
}

mod c_double_targ_consts {
    const radix: uint = 2u;
    const mantissa_digits: uint = 53u;
    const digits: uint = 15u;
    const min_exp: uint = -1021u;
    const max_exp: uint = 1024u;
    const min_10_exp: int = -307;
    const max_10_exp: int = 308;
    // FIXME this is wrong! replace with hexadecimal (%a) constants below
    const min_value: f64 = 2.225074e-308_f64;
    const max_value: f64 = 1.797693e+308_f64;
    const epsilon: f64 = 2.220446e-16_f64;
}

/*

FIXME use these once they can be parsed

mod c_float_math_consts {
    const pi: c_float = 0x1.921fb6p+1_f32;
    const div_1_pi: c_float = 0x1.45f306p-2_f32;
    const div_2_pi: c_float = 0x1.45f306p-1_f32;
    const div_pi_2: c_float = 0x1.921fb6p+0_f32;
    const div_pi_4: c_float = 0x1.921fb6p-1_f32;
    const div_2_sqrtpi: c_float = 0x1.20dd76p+0_f32;
    const e: c_float = 0x1.5bf0a8p+1_f32;
    const log2_e: c_float = 0x1.715476p+0_f32;
    const log10_e: c_float = 0x1.bcb7b2p-2_f32;
    const ln_2: c_float = 0x1.62e43p-1_f32;
    const ln_10: c_float = 0x1.26bb1cp+1_f32;
    const sqrt2: c_float = 0x1.6a09e6p+0_f32;
    const div_1_sqrt2: c_float = 0x1.6a09e6p-1_f32;
}

mod c_double_math_consts {
    const pi: c_double = 0x1.921fb54442d18p+1_f64;
    const div_1_pi: c_double = 0x1.45f306dc9c883p-2_f64;
    const div_2_pi: c_double = 0x1.45f306dc9c883p-1_f64;
    const div_pi_2: c_double = 0x1.921fb54442d18p+0_f64;
    const div_pi_4: c_double = 0x1.921fb54442d18p-1_f64;
    const div_2_sqrtpi: c_double = 0x1.20dd750429b6dp+0_f64;
    const e: c_double = 0x1.5bf0a8b145769p+1_f64;
    const log2_e: c_double = 0x1.71547652b82fep+0_f64;
    const log10_e: c_double = 0x1.bcb7b1526e50ep-2_f64;
    const ln_2: c_double = 0x1.62e42fefa39efp-1_f64;
    const ln_10: c_double = 0x1.26bb1bbb55516p+1_f64;
    const sqrt2: c_double = 0x1.6a09e667f3bcdp+0_f64;
    const div_1_sqrt2: c_double = 0x1.6a09e667f3bcdp-1_f64;
}

mod c_float_targ_consts {
    const radix: uint = 2u;
    const mantissa_digits: uint = 24u;
    const digits: uint = 6u;
    const min_exp: int = -125;
    const max_exp: int = 128;
    const min_10_exp: int = -37;
    const max_10_exp: int = 38;
    const min_value: c_float = 0x1p-126_f32;
    const max_value: c_float = 0x1.fffffep+127_f32;
    const epsilon: c_float = 0x1p-23_f32;
}

mod c_double_targ_consts {
    const radix: uint = 2u;
    const mantissa_digits: uint = 53u;
    const digits: uint = 15u;
    const min_exp: int = -1021;
    const max_exp: int = 1024;
    const min_10_exp: int = -307;
    const max_10_exp: int = 308;
    const min_value: c_double = 0x1p-1022_f64;
    const max_value: c_double = 0x1.fffffffffffffp+1023_f64;
    const epsilon: c_double = 0x1p-52_f64;
}

*/

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//


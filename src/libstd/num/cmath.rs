// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)]; // FIXME #3538

// function names are almost identical to C's libmath, a few have been
// renamed, grep for "rename:"

pub mod c_double_utils {
    use libc::{c_double, c_int};

    #[link_name = "m"]
    #[abi = "cdecl"]
    pub extern {
        // Alpabetically sorted by link_name

        unsafe fn acos(n: c_double) -> c_double;
        unsafe fn asin(n: c_double) -> c_double;
        unsafe fn atan(n: c_double) -> c_double;
        unsafe fn atan2(a: c_double, b: c_double) -> c_double;
        unsafe fn cbrt(n: c_double) -> c_double;
        unsafe fn ceil(n: c_double) -> c_double;
        unsafe fn copysign(x: c_double, y: c_double) -> c_double;
        unsafe fn cos(n: c_double) -> c_double;
        unsafe fn cosh(n: c_double) -> c_double;
        unsafe fn erf(n: c_double) -> c_double;
        unsafe fn erfc(n: c_double) -> c_double;
        unsafe fn exp(n: c_double) -> c_double;
        // rename: for consistency with underscore usage elsewhere
        #[link_name="expm1"] unsafe fn exp_m1(n: c_double) -> c_double;
        unsafe fn exp2(n: c_double) -> c_double;
        #[link_name="fabs"] unsafe fn abs(n: c_double) -> c_double;
        // rename: for clarity and consistency with add/sub/mul/div
        #[link_name="fdim"]
        unsafe fn abs_sub(a: c_double, b: c_double) -> c_double;
        unsafe fn floor(n: c_double) -> c_double;
        // rename: for clarity and consistency with add/sub/mul/div
        #[link_name="fma"]
        unsafe fn mul_add(a: c_double, b: c_double, c: c_double) -> c_double;
        #[link_name="fmax"]
        unsafe fn fmax(a: c_double, b: c_double) -> c_double;
        #[link_name="fmin"]
        unsafe fn fmin(a: c_double, b: c_double) -> c_double;
        #[link_name="nextafter"]
        unsafe fn next_after(x: c_double, y: c_double) -> c_double;
        unsafe fn frexp(n: c_double, value: &mut c_int) -> c_double;
        unsafe fn hypot(x: c_double, y: c_double) -> c_double;
        unsafe fn ldexp(x: c_double, n: c_int) -> c_double;
        #[cfg(unix)]
        #[link_name="lgamma_r"]
        unsafe fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        #[cfg(windows)]
        #[link_name="__lgamma_r"]
        unsafe fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        // renamed: log is a reserved keyword; ln seems more natural, too
        #[link_name="log"] unsafe fn ln(n: c_double) -> c_double;
        // renamed: "logb" /often/ is confused for log2 by beginners
        #[link_name="logb"] unsafe fn log_radix(n: c_double) -> c_double;
        // renamed: to be consitent with log as ln
        #[link_name="log1p"] unsafe fn ln_1p(n: c_double) -> c_double;
        unsafe fn log10(n: c_double) -> c_double;
        unsafe fn log2(n: c_double) -> c_double;
        #[link_name="ilogb"] unsafe fn ilog_radix(n: c_double) -> c_int;
        unsafe fn modf(n: c_double, iptr: &mut c_double) -> c_double;
        unsafe fn pow(n: c_double, e: c_double) -> c_double;
    // FIXME (#1379): enable when rounding modes become available
    //    unsafe fn rint(n: c_double) -> c_double;
        unsafe fn round(n: c_double) -> c_double;
        // rename: for consistency with logradix
        #[link_name="scalbn"] unsafe fn ldexp_radix(n: c_double, i: c_int) ->
            c_double;
        unsafe fn sin(n: c_double) -> c_double;
        unsafe fn sinh(n: c_double) -> c_double;
        unsafe fn sqrt(n: c_double) -> c_double;
        unsafe fn tan(n: c_double) -> c_double;
        unsafe fn tanh(n: c_double) -> c_double;
        unsafe fn tgamma(n: c_double) -> c_double;
        unsafe fn trunc(n: c_double) -> c_double;

        // These are commonly only available for doubles

        unsafe fn j0(n: c_double) -> c_double;
        unsafe fn j1(n: c_double) -> c_double;
        unsafe fn jn(i: c_int, n: c_double) -> c_double;

        unsafe fn y0(n: c_double) -> c_double;
        unsafe fn y1(n: c_double) -> c_double;
        unsafe fn yn(i: c_int, n: c_double) -> c_double;
    }
}

pub mod c_float_utils {
    use libc::{c_float, c_int};

    #[link_name = "m"]
    #[abi = "cdecl"]
    pub extern {
        // Alpabetically sorted by link_name

        #[link_name="acosf"] unsafe fn acos(n: c_float) -> c_float;
        #[link_name="asinf"] unsafe fn asin(n: c_float) -> c_float;
        #[link_name="atanf"] unsafe fn atan(n: c_float) -> c_float;
        #[link_name="atan2f"]
        unsafe fn atan2(a: c_float, b: c_float) -> c_float;
        #[link_name="cbrtf"] unsafe fn cbrt(n: c_float) -> c_float;
        #[link_name="ceilf"] unsafe fn ceil(n: c_float) -> c_float;
        #[link_name="copysignf"] unsafe fn copysign(x: c_float,
                                                  y: c_float) -> c_float;
        #[link_name="cosf"] unsafe fn cos(n: c_float) -> c_float;
        #[link_name="coshf"] unsafe fn cosh(n: c_float) -> c_float;
        #[link_name="erff"] unsafe fn erf(n: c_float) -> c_float;
        #[link_name="erfcf"] unsafe fn erfc(n: c_float) -> c_float;
        #[link_name="expf"] unsafe fn exp(n: c_float) -> c_float;
        #[link_name="expm1f"]unsafe fn exp_m1(n: c_float) -> c_float;
        #[link_name="exp2f"] unsafe fn exp2(n: c_float) -> c_float;
        #[link_name="fabsf"] unsafe fn abs(n: c_float) -> c_float;
        #[link_name="fdimf"]
        unsafe fn abs_sub(a: c_float, b: c_float) -> c_float;
        #[link_name="floorf"] unsafe fn floor(n: c_float) -> c_float;
        #[link_name="frexpf"] unsafe fn frexp(n: c_float,
                                            value: &mut c_int) -> c_float;
        #[link_name="fmaf"]
        unsafe fn mul_add(a: c_float, b: c_float, c: c_float) -> c_float;
        #[link_name="fmaxf"]
        unsafe fn fmax(a: c_float, b: c_float) -> c_float;
        #[link_name="fminf"]
        unsafe fn fmin(a: c_float, b: c_float) -> c_float;
        #[link_name="nextafterf"]
        unsafe fn next_after(x: c_float, y: c_float) -> c_float;
        #[link_name="hypotf"]
        unsafe fn hypot(x: c_float, y: c_float) -> c_float;
        #[link_name="ldexpf"]
        unsafe fn ldexp(x: c_float, n: c_int) -> c_float;

        #[cfg(unix)]
        #[link_name="lgammaf_r"] unsafe fn lgamma(n: c_float,
                                                sign: &mut c_int) -> c_float;

        #[cfg(windows)]
        #[link_name="__lgammaf_r"]
        unsafe fn lgamma(n: c_float, sign: &mut c_int) -> c_float;

        #[link_name="logf"] unsafe fn ln(n: c_float) -> c_float;
        #[link_name="logbf"] unsafe fn log_radix(n: c_float) -> c_float;
        #[link_name="log1pf"] unsafe fn ln_1p(n: c_float) -> c_float;
        #[link_name="log2f"] unsafe fn log2(n: c_float) -> c_float;
        #[link_name="log10f"] unsafe fn log10(n: c_float) -> c_float;
        #[link_name="ilogbf"] unsafe fn ilog_radix(n: c_float) -> c_int;
        #[link_name="modff"] unsafe fn modf(n: c_float,
                                          iptr: &mut c_float) -> c_float;
        #[link_name="powf"] unsafe fn pow(n: c_float, e: c_float) -> c_float;
    // FIXME (#1379): enable when rounding modes become available
    //    #[link_name="rintf"] unsafe fn rint(n: c_float) -> c_float;
        #[link_name="roundf"] unsafe fn round(n: c_float) -> c_float;
        #[link_name="scalbnf"] unsafe fn ldexp_radix(n: c_float, i: c_int)
            -> c_float;
        #[link_name="sinf"] unsafe fn sin(n: c_float) -> c_float;
        #[link_name="sinhf"] unsafe fn sinh(n: c_float) -> c_float;
        #[link_name="sqrtf"] unsafe fn sqrt(n: c_float) -> c_float;
        #[link_name="tanf"] unsafe fn tan(n: c_float) -> c_float;
        #[link_name="tanhf"] unsafe fn tanh(n: c_float) -> c_float;
        #[link_name="tgammaf"] unsafe fn tgamma(n: c_float) -> c_float;
        #[link_name="truncf"] unsafe fn trunc(n: c_float) -> c_float;
    }
}

// PORT check these by running src/etc/machconsts.c for your architecture

// FIXME obtain machine float/math constants automatically (Issue #1986)

pub mod c_float_targ_consts {
    pub static radix: uint = 2u;
    pub static mantissa_digits: uint = 24u;
    pub static digits: uint = 6u;
    pub static min_exp: uint = -125u;
    pub static max_exp: uint = 128u;
    pub static min_10_exp: int = -37;
    pub static max_10_exp: int = 38;
    // FIXME (#1433): this is wrong, replace with hexadecimal (%a) staticants
    // below.
    pub static min_value: f32 = 1.175494e-38_f32;
    pub static max_value: f32 = 3.402823e+38_f32;
    pub static epsilon: f32 = 0.000000_f32;
}

pub mod c_double_targ_consts {
    pub static radix: uint = 2u;
    pub static mantissa_digits: uint = 53u;
    pub static digits: uint = 15u;
    pub static min_exp: uint = -1021u;
    pub static max_exp: uint = 1024u;
    pub static min_10_exp: int = -307;
    pub static max_10_exp: int = 308;
    // FIXME (#1433): this is wrong, replace with hexadecimal (%a) staticants
    // below.
    pub static min_value: f64 = 2.225074e-308_f64;
    pub static max_value: f64 = 1.797693e+308_f64;
    pub static epsilon: f64 = 2.220446e-16_f64;
}

/*

FIXME use these once they can be parsed (see Issue #1433)

pub mod c_float_math_consts {
    pub static pi: c_float = 0x1.921fb6p+1_f32;
    pub static div_1_pi: c_float = 0x1.45f306p-2_f32;
    pub static div_2_pi: c_float = 0x1.45f306p-1_f32;
    pub static div_pi_2: c_float = 0x1.921fb6p+0_f32;
    pub static div_pi_4: c_float = 0x1.921fb6p-1_f32;
    pub static div_2_sqrtpi: c_float = 0x1.20dd76p+0_f32;
    pub static e: c_float = 0x1.5bf0a8p+1_f32;
    pub static log2_e: c_float = 0x1.715476p+0_f32;
    pub static log10_e: c_float = 0x1.bcb7b2p-2_f32;
    pub static ln_2: c_float = 0x1.62e43p-1_f32;
    pub static ln_10: c_float = 0x1.26bb1cp+1_f32;
    pub static sqrt2: c_float = 0x1.6a09e6p+0_f32;
    pub static div_1_sqrt2: c_float = 0x1.6a09e6p-1_f32;
}

pub mod c_double_math_consts {
    pub static pi: c_double = 0x1.921fb54442d18p+1_f64;
    pub static div_1_pi: c_double = 0x1.45f306dc9c883p-2_f64;
    pub static div_2_pi: c_double = 0x1.45f306dc9c883p-1_f64;
    pub static div_pi_2: c_double = 0x1.921fb54442d18p+0_f64;
    pub static div_pi_4: c_double = 0x1.921fb54442d18p-1_f64;
    pub static div_2_sqrtpi: c_double = 0x1.20dd750429b6dp+0_f64;
    pub static e: c_double = 0x1.5bf0a8b145769p+1_f64;
    pub static log2_e: c_double = 0x1.71547652b82fep+0_f64;
    pub static log10_e: c_double = 0x1.bcb7b1526e50ep-2_f64;
    pub static ln_2: c_double = 0x1.62e42fefa39efp-1_f64;
    pub static ln_10: c_double = 0x1.26bb1bbb55516p+1_f64;
    pub static sqrt2: c_double = 0x1.6a09e667f3bcdp+0_f64;
    pub static div_1_sqrt2: c_double = 0x1.6a09e667f3bcdp-1_f64;
}

pub mod c_float_targ_consts {
    pub static radix: uint = 2u;
    pub static mantissa_digits: uint = 24u;
    pub static digits: uint = 6u;
    pub static min_exp: int = -125;
    pub static max_exp: int = 128;
    pub static min_10_exp: int = -37;
    pub static max_10_exp: int = 38;
    pub static min_value: c_float = 0x1p-126_f32;
    pub static max_value: c_float = 0x1.fffffep+127_f32;
    pub static epsilon: c_float = 0x1p-23_f32;
}

pub mod c_double_targ_consts {
    pub static radix: uint = 2u;
    pub static mantissa_digits: uint = 53u;
    pub static digits: uint = 15u;
    pub static min_exp: int = -1021;
    pub static max_exp: int = 1024;
    pub static min_10_exp: int = -307;
    pub static max_10_exp: int = 308;
    pub static min_value: c_double = 0x1p-1022_f64;
    pub static max_value: c_double = 0x1.fffffffffffffp+1023_f64;
    pub static epsilon: c_double = 0x1p-52_f64;
}

*/

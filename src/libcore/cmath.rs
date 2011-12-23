export c_double;
export c_float;
export bessel;

import ctypes::c_int;
import ctypes::c_float;
import ctypes::c_double;

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
    #[link_name="fdim"] pure fn sub_pos(a: c_double, b: c_double) -> c_double;
    pure fn floor(n: c_double) -> c_double;
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
    #[link_name="log"] pure fn ln(n: c_double) -> c_double;
    pure fn logb(n: c_double) -> c_double;
    #[link_name="log1p"] pure fn ln1p(n: c_double) -> c_double;
    pure fn log10(n: c_double) -> c_double;
    pure fn log2(n: c_double) -> c_double;
    pure fn ilogb(n: c_double) -> c_int;
    pure fn modf(n: c_double, &iptr: c_double) -> c_double;
    pure fn pow(n: c_double, e: c_double) -> c_double;
    pure fn rint(n: c_double) -> c_double;
    pure fn round(n: c_double) -> c_double;
    pure fn scalbn(n: c_double, i: c_int) -> c_double;
    pure fn sin(n: c_double) -> c_double;
    pure fn sinh(n: c_double) -> c_double;
    pure fn sqrt(n: c_double) -> c_double;
    pure fn tan(n: c_double) -> c_double;
    pure fn tanh(n: c_double) -> c_double;
    pure fn tgamma(n: c_double) -> c_double;
    pure fn trunc(n: c_double) -> c_double;
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
    #[link_name="fdimf"] pure fn sub_pos(a: c_float, b: c_float) -> c_float;
    #[link_name="floorf"] pure fn floor(n: c_float) -> c_float;
    #[link_name="frexpf"] pure fn frexp(n: c_double,
                                        &value: c_int) -> c_float;
    #[link_name="fmaf"] pure fn mul_add(a: c_float,
                                        b: c_float, c: c_float) -> c_float;
    #[link_name="fmaxf"] pure fn fmax(a: c_float, b: c_float) -> c_float;
    #[link_name="fminf"] pure fn fmin(a: c_float, b: c_float) -> c_float;
    #[link_name="nextafterf"] pure fn nextafter(x: c_float,
                                                y: c_float) -> c_float;
    #[link_name="hypotf"] pure fn hypot(x: c_float, y: c_float) -> c_float;
    #[link_name="ldexpf"] pure fn ldexp(x: c_float, n: c_int) -> c_float;
    #[link_name="lgammaf_r"] pure fn lgamma(n: c_float,
                                            &sign: c_int) -> c_float;
    #[link_name="logf"] pure fn ln(n: c_float) -> c_float;
    #[link_name="logbf"] pure fn logb(n: c_float) -> c_float;
    #[link_name="log1p"] pure fn ln1p(n: c_double) -> c_double;
    #[link_name="log2f"] pure fn log2(n: c_float) -> c_float;
    #[link_name="log10f"] pure fn log10(n: c_float) -> c_float;
    #[link_name="ilogbf"] pure fn ilogb(n: c_float) -> c_int;
    #[link_name="modff"] pure fn modf(n: c_float,
                                      &iptr: c_float) -> c_float;
    #[link_name="powf"] pure fn pow(n: c_float, e: c_float) -> c_float;
    #[link_name="rintf"] pure fn rint(n: c_float) -> c_float;
    #[link_name="roundf"] pure fn round(n: c_float) -> c_float;
    #[link_name="scalbnf"] pure fn scalbn(n: c_float, i: c_int) -> c_float;
    #[link_name="sinf"] pure fn sin(n: c_float) -> c_float;
    #[link_name="sinhf"] pure fn sinh(n: c_float) -> c_float;
    #[link_name="sqrtf"] pure fn sqrt(n: c_float) -> c_float;
    #[link_name="tanf"] pure fn tan(n: c_float) -> c_float;
    #[link_name="tanhf"] pure fn tanh(n: c_float) -> c_float;
    #[link_name="tgammaf"] pure fn tgamma(n: c_float) -> c_float;
    #[link_name="truncf"] pure fn trunc(n: c_float) -> c_float;
}

#[link_name = "m"]
#[abi = "cdecl"]
native mod bessel {
    pure fn j0(n: c_double) -> c_double;
    pure fn j1(n: c_double) -> c_double;
    pure fn jn(i: c_int, n: c_double) -> c_double;

    pure fn y0(n: c_double) -> c_double;
    pure fn y1(n: c_double) -> c_double;
    pure fn yn(i: c_int, n: c_double) -> c_double;
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


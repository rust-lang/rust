import ctypes::c_int;

#[link_name = "m"]
#[abi = "cdecl"]
native mod f64 {

    // Alpabetically sorted by link_name

    pure fn acos(n: f64) -> f64;
    pure fn asin(n: f64) -> f64;
    pure fn atan(n: f64) -> f64;
    pure fn atan2(a: f64, b: f64) -> f64;
    pure fn cbrt(n: f64) -> f64;
    pure fn ceil(n: f64) -> f64;
    pure fn cos(n: f64) -> f64;
    pure fn cosh(n: f64) -> f64;
    pure fn erf(n: f64) -> f64;
    pure fn erfc(n: f64) -> f64;
    pure fn exp(n: f64) -> f64;
    pure fn expm1(n: f64) -> f64;
    pure fn exp2(n: f64) -> f64;
    #[link_name="fabs"] pure fn abs(n: f64) -> f64;
    #[link_name="fdim"] pure fn sub_pos(a: f64, b: f64) -> f64;
    pure fn floor(n: f64) -> f64;
    #[link_name="fma"] pure fn mul_add(a: f64, b: f64, c: f64) -> f64;
    #[link_name="fmax"] pure fn max(a: f64, b: f64) -> f64;
    #[link_name="fmin"] pure fn min(a: f64, b: f64) -> f64;
    pure fn nextafter(x: f64, y: f64) -> f64
    pure fn fmod(x: f64, y: f64) -> f64;
    pure fn frexp(n: f64, &value: c_int) -> f64;
    pure fn hypot(x: f64, y: f64) -> f64;
    pure fn ldexp(x: f64, n: c_int) -> f64;
    #[link_name="lgamma_r"] pure fn lgamma(n: f64, sign: *c_int) -> f64;
    #[link_name="log"] pure fn ln(n: f64) -> f64;
    pure fn logb(n: f64) -> f64;
    #[link_name="log1p"] pure fn ln1p(n: f64) -> f64;
    pure fn log10(n: f64) -> f64;
    pure fn log2(n: f64) -> f64;
    pure fn ilogb(n: f64) -> c_int;
    pure fn modf(n: f64, iptr: *f64) -> f64;
    pure fn pow(n: f64, e: f64) -> f64;
    #[link_name="remainder"] pure fn rem(a: f64, b: f64) -> f64;
    pure fn rint(n: f64) -> f64;
    pure fn round(n: f64) -> f64;
    pure fn sin(n: f64) -> f64;
    pure fn sinh(n: f64) -> f64;
    pure fn sqrt(n: f64) -> f64;
    pure fn tan(n: f64) -> f64;
    pure fn tanh(n: f64) -> f64;
    pure fn tgamma(n: f64) -> f64;
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
    #[link_name="cbrtf"] pure fn cbrt(n: f32) -> f32;
    #[link_name="ceilf"] pure fn ceil(n: f32) -> f32;
    #[link_name="cosf"] pure fn cos(n: f32) -> f32;
    #[link_name="coshf"] pure fn cosh(n: f32) -> f32;
    #[link_name="erff"] pure fn erf(n: f32) -> f32;
    #[link_name="erfcf"] pure fn erfc(n: f32) -> f32;
    #[link_name="expf"] pure fn exp(n: f32) -> f32;
    #[link_name="expm1f"]pure fn expm1(n: f32) -> f32;
    #[link_name="exp2f"] pure fn exp2(n: f32) -> f32;
    #[link_name="fabsf"] pure fn abs(n: f32) -> f32;
    #[link_name="fdimf"] pure fn sub_pos(a: f32, b: f32) -> f32;
    #[link_name="floorf"] pure fn floor(n: f32) -> f32;
    #[link_name="frexpf"] pure fn frexp(n: f64, &value: c_int) -> f32;
    #[link_name="fmaf"] pure fn mul_add(a: f32, b: f32, c: f32) -> f32;
    #[link_name="fmaxf"] pure fn max(a: f32, b: f32) -> f32;
    #[link_name="fminf"] pure fn min(a: f32, b: f32) -> f32;
    #[link_name="nextafterf"] pure fn nextafter(x: f32, y: f32) -> f32
    #[link_name="fmodf"] pure fn fmod(x: f32, y: f32) -> f32;
    #[link_name="hypotf"] pure fn hypot(x: f32, y: f32) -> f32;
    #[link_name="ldexpf"] pure fn ldexp(x: f32, n: c_int) -> f32;
    #[link_name="lgammaf_r"] pure fn lgamma(n: f32, sign: *c_int) -> f32;
    #[link_name="logf"] pure fn ln(n: f32) -> f32;
    #[link_name="logbf"] pure fn logb(n: f32) -> f32;
    #[link_name="log1p"] pure fn ln1p(n: f64) -> f64;
    #[link_name="log2f"] pure fn log2(n: f32) -> f32;
    #[link_name="log10f"] pure fn log10(n: f32) -> f32;
    #[link_name="ilogbf"] pure fn ilogb(n: f32) -> c_int;
    #[link_name="modff"] pure fn modf(n: f32, iptr: *f32) -> f32;
    #[link_name="powf"] pure fn pow(n: f32, e: f32) -> f32;
    #[link_name="remainderf"] pure fn rem(a: f32, b: f32) -> f32;
    #[link_name="rintf"] pure fn rint(n: f32) -> f32;
    #[link_name="roundf"] pure fn round(n: f32) -> f32;
    #[link_name="sinf"] pure fn sin(n: f32) -> f32;
    #[link_name="sinhf"] pure fn sinh(n: f32) -> f32;
    #[link_name="sqrtf"] pure fn sqrt(n: f32) -> f32;
    #[link_name="tanf"] pure fn tan(n: f32) -> f32;
    #[link_name="tanhf"] pure fn tanh(n: f32) -> f32;
    #[link_name="tgammaf"] pure fn tgamma(n: f32) -> f32;
    #[link_name="truncf"] pure fn trunc(n: f32) -> f32;
}

#[link_name = "m"]
#[abi = "cdecl"]
native mod bessel {
    pure fn j0(n: m_float) -> m_float;
    pure fn j1(n: m_float) -> m_float;
    pure fn jn(i: c_int, n: m_float) -> m_float;

    pure fn y0(n: m_float) -> m_float;
    pure fn y1(n: m_float) -> m_float;
    pure fn yn(i: c_int, n: m_float) -> m_float;
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


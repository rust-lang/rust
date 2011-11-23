import ctypes::c_int;

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

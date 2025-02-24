use core::ffi::c_int;

#[allow(dead_code)]
#[allow(clippy::all)] // We don't get `libm`'s list of `allow`s, so just ignore Clippy.
#[allow(unused_imports)]
#[path = "../../../src/math/mod.rs"]
pub mod libm;

/// Mark functions `#[no_mangle]` and with the C ABI.
macro_rules! no_mangle {
    ($( $name:ident( $($tt:tt)+ ) -> $ret:ty; )+) => {
        $( no_mangle!(@inner $name( $($tt)+ ) -> $ret); )+
    };

    // Handle simple functions with single return types
    (@inner $name:ident( $($arg:ident: $aty:ty),+ ) -> $ret:ty) => {
        #[no_mangle]
        extern "C" fn $name($($arg: $aty),+) -> $ret {
            libm::$name($($arg),+)
        }
    };


    // Functions with `&mut` return values need to be handled differently, use `|` to
    // separate inputs vs. outputs.
    (
        @inner $name:ident( $($arg:ident: $aty:ty),+ | $($rarg:ident: $rty:ty),+) -> $ret:ty
    ) => {
        #[no_mangle]
        extern "C" fn $name($($arg: $aty,)+ $($rarg: $rty),+) -> $ret {
            let ret;
            (ret, $(*$rarg),+) = libm::$name($($arg),+);
            ret
        }
    };
}

no_mangle! {
    frexp(x: f64 | y: &mut c_int) -> f64;
    frexpf(x: f32 | y: &mut c_int) -> f32;
    acos(x: f64) -> f64;
    acosf(x: f32) -> f32;
    acosh(x: f64) -> f64;
    acoshf(x: f32) -> f32;
    asin(x: f64) -> f64;
    asinf(x: f32) -> f32;
    asinh(x: f64) -> f64;
    asinhf(x: f32) -> f32;
    atan(x: f64) -> f64;
    atan2(x: f64, y: f64) -> f64;
    atan2f(x: f32, y: f32) -> f32;
    atanf(x: f32) -> f32;
    atanh(x: f64) -> f64;
    atanhf(x: f32) -> f32;
    cbrt(x: f64) -> f64;
    cbrtf(x: f32) -> f32;
    ceil(x: f64) -> f64;
    ceilf(x: f32) -> f32;
    ceilf128(x: f128) -> f128;
    ceilf16(x: f16) -> f16;
    copysign(x: f64, y: f64) -> f64;
    copysignf(x: f32, y: f32) -> f32;
    copysignf128(x: f128, y: f128) -> f128;
    copysignf16(x: f16, y: f16) -> f16;
    cos(x: f64) -> f64;
    cosf(x: f32) -> f32;
    cosh(x: f64) -> f64;
    coshf(x: f32) -> f32;
    erf(x: f64) -> f64;
    erfc(x: f64) -> f64;
    erfcf(x: f32) -> f32;
    erff(x: f32) -> f32;
    exp(x: f64) -> f64;
    exp10(x: f64) -> f64;
    exp10f(x: f32) -> f32;
    exp2(x: f64) -> f64;
    exp2f(x: f32) -> f32;
    expf(x: f32) -> f32;
    expm1(x: f64) -> f64;
    expm1f(x: f32) -> f32;
    fabs(x: f64) -> f64;
    fabsf(x: f32) -> f32;
    fabsf128(x: f128) -> f128;
    fabsf16(x: f16) -> f16;
    fdim(x: f64, y: f64) -> f64;
    fdimf(x: f32, y: f32) -> f32;
    fdimf128(x: f128, y: f128) -> f128;
    fdimf16(x: f16, y: f16) -> f16;
    floor(x: f64) -> f64;
    floorf(x: f32) -> f32;
    floorf128(x: f128) -> f128;
    floorf16(x: f16) -> f16;
    fma(x: f64, y: f64, z: f64) -> f64;
    fmaf(x: f32, y: f32, z: f32) -> f32;
    fmax(x: f64, y: f64) -> f64;
    fmaxf(x: f32, y: f32) -> f32;
    fmin(x: f64, y: f64) -> f64;
    fminf(x: f32, y: f32) -> f32;
    fmod(x: f64, y: f64) -> f64;
    fmodf(x: f32, y: f32) -> f32;
    hypot(x: f64, y: f64) -> f64;
    hypotf(x: f32, y: f32) -> f32;
    ilogb(x: f64) -> c_int;
    ilogbf(x: f32) -> c_int;
    j0(x: f64) -> f64;
    j0f(x: f32) -> f32;
    j1(x: f64) -> f64;
    j1f(x: f32) -> f32;
    jn(x: c_int, y: f64) -> f64;
    jnf(x: c_int, y: f32) -> f32;
    ldexp(x: f64, y: c_int) -> f64;
    ldexpf(x: f32, y: c_int) -> f32;
    lgamma(x: f64) -> f64;
    lgamma_r(x: f64 | r: &mut c_int) -> f64;
    lgammaf(x: f32) -> f32;
    lgammaf_r(x: f32 | r: &mut c_int) -> f32;
    log(x: f64) -> f64;
    log10(x: f64) -> f64;
    log10f(x: f32) -> f32;
    log1p(x: f64) -> f64;
    log1pf(x: f32) -> f32;
    log2(x: f64) -> f64;
    log2f(x: f32) -> f32;
    logf(x: f32) -> f32;
    modf(x: f64 | r: &mut f64) -> f64;
    modff(x: f32 | r: &mut f32) -> f32;
    nextafter(x: f64, y: f64) -> f64;
    nextafterf(x: f32, y: f32) -> f32;
    pow(x: f64, y: f64) -> f64;
    powf(x: f32, y: f32) -> f32;
    remainder(x: f64, y: f64) -> f64;
    remainderf(x: f32, y: f32) -> f32;
    remquo(x: f64, y: f64 | q: &mut c_int) -> f64;
    remquof(x: f32, y: f32 | q: &mut c_int) -> f32;
    rint(x: f64) -> f64;
    rintf(x: f32) -> f32;
    rintf128(x: f128) -> f128;
    rintf16(x: f16) -> f16;
    round(x: f64) -> f64;
    roundf(x: f32) -> f32;
    scalbn(x: f64, y: c_int) -> f64;
    scalbnf(x: f32, y: c_int) -> f32;
    sin(x: f64) -> f64;
    sinf(x: f32) -> f32;
    sinh(x: f64) -> f64;
    sinhf(x: f32) -> f32;
    sqrt(x: f64) -> f64;
    sqrtf(x: f32) -> f32;
    tan(x: f64) -> f64;
    tanf(x: f32) -> f32;
    tanh(x: f64) -> f64;
    tanhf(x: f32) -> f32;
    tgamma(x: f64) -> f64;
    tgammaf(x: f32) -> f32;
    trunc(x: f64) -> f64;
    truncf(x: f32) -> f32;
    truncf128(x: f128) -> f128;
    truncf16(x: f16) -> f16;
    y0(x: f64) -> f64;
    y0f(x: f32) -> f32;
    y1(x: f64) -> f64;
    y1f(x: f32) -> f32;
    yn(x: c_int, y: f64) -> f64;
    ynf(x: c_int, y: f32) -> f32;
}

/* sincos has no direct return type, not worth handling in the macro */

#[no_mangle]
extern "C" fn sincos(x: f64, s: &mut f64, c: &mut f64) {
    (*s, *c) = libm::sincos(x);
}

#[no_mangle]
extern "C" fn sincosf(x: f32, s: &mut f32, c: &mut f32) {
    (*s, *c) = libm::sincosf(x);
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

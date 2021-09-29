#[allow(dead_code)]
#[path = "../libm/src/math/mod.rs"]
mod libm;

macro_rules! no_mangle {
    ($(fn $fun:ident($($iid:ident : $ity:ty),+) -> $oty:ty;)+) => {
        intrinsics! {
            $(
                pub extern "C" fn $fun($($iid: $ity),+) -> $oty {
                    self::libm::$fun($($iid),+)
                }
            )+
        }
    }
}

#[cfg(any(
    all(
        target_arch = "wasm32",
        target_os = "unknown",
        not(target_env = "wasi")
    ),
    all(target_arch = "x86_64", target_os = "uefi"),
    all(target_vendor = "fortanix", target_env = "sgx")
))]
no_mangle! {
    fn acos(x: f64) -> f64;
    fn asin(x: f64) -> f64;
    fn cbrt(x: f64) -> f64;
    fn expm1(x: f64) -> f64;
    fn hypot(x: f64, y: f64) -> f64;
    fn tan(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn expf(x: f32) -> f32;
    fn log2(x: f64) -> f64;
    fn log2f(x: f32) -> f32;
    fn log10(x: f64) -> f64;
    fn log10f(x: f32) -> f32;
    fn log(x: f64) -> f64;
    fn logf(x: f32) -> f32;
    fn fmin(x: f64, y: f64) -> f64;
    fn fminf(x: f32, y: f32) -> f32;
    fn fmax(x: f64, y: f64) -> f64;
    fn fmaxf(x: f32, y: f32) -> f32;
    fn round(x: f64) -> f64;
    fn roundf(x: f32) -> f32;
    fn sin(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn powf(x: f32, y: f32) -> f32;
    fn fmod(x: f64, y: f64) -> f64;
    fn fmodf(x: f32, y: f32) -> f32;
    fn acosf(n: f32) -> f32;
    fn atan2f(a: f32, b: f32) -> f32;
    fn atanf(n: f32) -> f32;
    fn coshf(n: f32) -> f32;
    fn expm1f(n: f32) -> f32;
    fn fdim(a: f64, b: f64) -> f64;
    fn fdimf(a: f32, b: f32) -> f32;
    fn log1pf(n: f32) -> f32;
    fn sinhf(n: f32) -> f32;
    fn tanhf(n: f32) -> f32;
    fn ldexp(f: f64, n: i32) -> f64;
    fn ldexpf(f: f32, n: i32) -> f32;
}

#[cfg(any(
    all(
        target_arch = "wasm32",
        target_os = "unknown",
        not(target_env = "wasi")
    ),
    all(target_vendor = "fortanix", target_env = "sgx")
))]
no_mangle! {
    fn atan(x: f64) -> f64;
    fn atan2(x: f64, y: f64) -> f64;
    fn cosh(x: f64) -> f64;
    fn log1p(x: f64) -> f64;
    fn sinh(x: f64) -> f64;
    fn tanh(x: f64) -> f64;
    fn cosf(x: f32) -> f32;
    fn exp(x: f64) -> f64;
    fn sinf(x: f32) -> f32;
    fn exp2(x: f64) -> f64;
    fn exp2f(x: f32) -> f32;
    fn fma(x: f64, y: f64, z: f64) -> f64;
    fn fmaf(x: f32, y: f32, z: f32) -> f32;
    fn asinf(n: f32) -> f32;
    fn cbrtf(n: f32) -> f32;
    fn hypotf(x: f32, y: f32) -> f32;
    fn tanf(n: f32) -> f32;
}

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
no_mangle! {
    fn ceil(x: f64) -> f64;
    fn ceilf(x: f32) -> f32;
    fn floor(x: f64) -> f64;
    fn floorf(x: f32) -> f32;
    fn trunc(x: f64) -> f64;
    fn truncf(x: f32) -> f32;
}

// only for the thumb*-none-eabi* targets
#[cfg(all(target_arch = "arm", target_os = "none"))]
no_mangle! {
    fn fmin(x: f64, y: f64) -> f64;
    fn fminf(x: f32, y: f32) -> f32;
    fn fmax(x: f64, y: f64) -> f64;
    fn fmaxf(x: f32, y: f32) -> f32;
    // `f64 % f64`
    fn fmod(x: f64, y: f64) -> f64;
    // `f32 % f32`
    fn fmodf(x: f32, y: f32) -> f32;
}

#[allow(dead_code)]
#[path = "../libm/src/math/mod.rs"]
mod libm;

macro_rules! no_mangle {
    ($(fn $fun:ident($($iid:ident : $ity:ty),+) -> $oty:ty;)+) => {
        $(
            #[no_mangle]
            pub fn $fun($($iid: $ity),+) -> $oty {
                self::libm::$fun($($iid),+)
            }
        )+
    }
}

no_mangle! {
    fn acos(x: f64) -> f64;
    fn asin(x: f64) -> f64;
    fn atan(x: f64) -> f64;
    fn atan2(x: f64, y: f64) -> f64;
    fn cbrt(x: f64) -> f64;
    fn cosh(x: f64) -> f64;
    fn expm1(x: f64) -> f64;
    fn hypot(x: f64, y: f64) -> f64;
    fn log1p(x: f64) -> f64;
    fn sinh(x: f64) -> f64;
    fn tan(x: f64) -> f64;
    fn tanh(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn cosf(x: f32) -> f32;
    fn exp(x: f64) -> f64;
    fn expf(x: f32) -> f32;
    fn log2(x: f64) -> f64;
    fn log2f(x: f32) -> f32;
    fn log10(x: f64) -> f64;
    fn log10f(x: f32) -> f32;
    fn log(x: f64) -> f64;
    fn logf(x: f32) -> f32;
    fn round(x: f64) -> f64;
    fn roundf(x: f32) -> f32;
    fn sin(x: f64) -> f64;
    fn sinf(x: f32) -> f32;
    fn pow(x: f64, y: f64) -> f64;
    fn powf(x: f32, y: f32) -> f32;
    fn exp2(x: f64) -> f64;
    fn exp2f(x: f32) -> f32;
    fn fmod(x: f64, y: f64) -> f64;
    fn fmodf(x: f32, y: f32) -> f32;
    fn fma(x: f64, y: f64, z: f64) -> f64;
    fn fmaf(x: f32, y: f32, z: f32) -> f32;
}

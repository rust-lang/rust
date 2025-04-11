//! Bindings to math functions provided by the system `libm` or by the `libm` crate, exposed
//! via `compiler-builtins`.

// These symbols are all defined by `libm`, or by `compiler-builtins` on unsupported platforms.
unsafe extern "C" {
    pub(crate) fn cbrt(n: f64) -> f64;
    pub(crate) fn cbrtf(n: f32) -> f32;
    pub(crate) fn fdim(a: f64, b: f64) -> f64;
    pub(crate) fn fdimf(a: f32, b: f32) -> f32;
}

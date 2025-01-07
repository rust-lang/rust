//! Wasm asm is not stable; just use intrinsics for operations that have asm routine equivalents.
//!
//! Note that we need to be absolutely certain that everything here lowers to assembly operations,
//! otherwise libcalls will be recursive.

pub fn ceil(x: f64) -> f64 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::ceilf64(x) }
}

pub fn ceilf(x: f32) -> f32 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::ceilf32(x) }
}

pub fn fabs(x: f64) -> f64 {
    x.abs()
}

pub fn fabsf(x: f32) -> f32 {
    x.abs()
}

pub fn floor(x: f64) -> f64 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::floorf64(x) }
}

pub fn floorf(x: f32) -> f32 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::floorf32(x) }
}

pub fn sqrt(x: f64) -> f64 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::sqrtf64(x) }
}

pub fn sqrtf(x: f32) -> f32 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::sqrtf32(x) }
}

pub fn trunc(x: f64) -> f64 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::truncf64(x) }
}

pub fn truncf(x: f32) -> f32 {
    // SAFETY: safe intrinsic with no preconditions
    unsafe { core::intrinsics::truncf32(x) }
}

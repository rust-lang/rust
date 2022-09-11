#![allow(non_camel_case_types)]

use crate::simd::Simd;

/// A 64-bit SIMD vector with two elements of type `f32`.
pub type f32x2 = Simd<f32, 2>;

/// A 128-bit SIMD vector with four elements of type `f32`.
pub type f32x4 = Simd<f32, 4>;

/// A 256-bit SIMD vector with eight elements of type `f32`.
pub type f32x8 = Simd<f32, 8>;

/// A 512-bit SIMD vector with 16 elements of type `f32`.
pub type f32x16 = Simd<f32, 16>;

/// A 128-bit SIMD vector with two elements of type `f64`.
pub type f64x2 = Simd<f64, 2>;

/// A 256-bit SIMD vector with four elements of type `f64`.
pub type f64x4 = Simd<f64, 4>;

/// A 512-bit SIMD vector with eight elements of type `f64`.
pub type f64x8 = Simd<f64, 8>;

#![allow(non_camel_case_types)]

use crate::simd::Simd;

/// A SIMD vector with two elements of type `usize`.
pub type usizex2 = Simd<usize, 2>;

/// A SIMD vector with four elements of type `usize`.
pub type usizex4 = Simd<usize, 4>;

/// A SIMD vector with eight elements of type `usize`.
pub type usizex8 = Simd<usize, 8>;

/// A 32-bit SIMD vector with two elements of type `u16`.
pub type u16x2 = Simd<u16, 2>;

/// A 64-bit SIMD vector with four elements of type `u16`.
pub type u16x4 = Simd<u16, 4>;

/// A 128-bit SIMD vector with eight elements of type `u16`.
pub type u16x8 = Simd<u16, 8>;

/// A 256-bit SIMD vector with 16 elements of type `u16`.
pub type u16x16 = Simd<u16, 16>;

/// A 512-bit SIMD vector with 32 elements of type `u16`.
pub type u16x32 = Simd<u16, 32>;

/// A 64-bit SIMD vector with two elements of type `u32`.
pub type u32x2 = Simd<u32, 2>;

/// A 128-bit SIMD vector with four elements of type `u32`.
pub type u32x4 = Simd<u32, 4>;

/// A 256-bit SIMD vector with eight elements of type `u32`.
pub type u32x8 = Simd<u32, 8>;

/// A 512-bit SIMD vector with 16 elements of type `u32`.
pub type u32x16 = Simd<u32, 16>;

/// A 128-bit SIMD vector with two elements of type `u64`.
pub type u64x2 = Simd<u64, 2>;

/// A 256-bit SIMD vector with four elements of type `u64`.
pub type u64x4 = Simd<u64, 4>;

/// A 512-bit SIMD vector with eight elements of type `u64`.
pub type u64x8 = Simd<u64, 8>;

/// A 32-bit SIMD vector with four elements of type `u8`.
pub type u8x4 = Simd<u8, 4>;

/// A 64-bit SIMD vector with eight elements of type `u8`.
pub type u8x8 = Simd<u8, 8>;

/// A 128-bit SIMD vector with 16 elements of type `u8`.
pub type u8x16 = Simd<u8, 16>;

/// A 256-bit SIMD vector with 32 elements of type `u8`.
pub type u8x32 = Simd<u8, 32>;

/// A 512-bit SIMD vector with 64 elements of type `u8`.
pub type u8x64 = Simd<u8, 64>;

#![allow(non_camel_case_types)]

use crate::simd::Simd;

/// Vector of two `usize` values
pub type usizex2 = Simd<usize, 2>;

/// Vector of four `usize` values
pub type usizex4 = Simd<usize, 4>;

/// Vector of eight `usize` values
pub type usizex8 = Simd<usize, 8>;

/// Vector of two `u16` values
pub type u16x2 = Simd<u16, 2>;

/// Vector of four `u16` values
pub type u16x4 = Simd<u16, 4>;

/// Vector of eight `u16` values
pub type u16x8 = Simd<u16, 8>;

/// Vector of 16 `u16` values
pub type u16x16 = Simd<u16, 16>;

/// Vector of 32 `u16` values
pub type u16x32 = Simd<u16, 32>;

/// Vector of two `u32` values
pub type u32x2 = Simd<u32, 2>;

/// Vector of four `u32` values
pub type u32x4 = Simd<u32, 4>;

/// Vector of eight `u32` values
pub type u32x8 = Simd<u32, 8>;

/// Vector of 16 `u32` values
pub type u32x16 = Simd<u32, 16>;

/// Vector of two `u64` values
pub type u64x2 = Simd<u64, 2>;

/// Vector of four `u64` values
pub type u64x4 = Simd<u64, 4>;

/// Vector of eight `u64` values
pub type u64x8 = Simd<u64, 8>;

/// Vector of four `u8` values
pub type u8x4 = Simd<u8, 4>;

/// Vector of eight `u8` values
pub type u8x8 = Simd<u8, 8>;

/// Vector of 16 `u8` values
pub type u8x16 = Simd<u8, 16>;

/// Vector of 32 `u8` values
pub type u8x32 = Simd<u8, 32>;

/// Vector of 64 `u8` values
pub type u8x64 = Simd<u8, 64>;

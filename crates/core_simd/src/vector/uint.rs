#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `u8` values.
pub type SimdU8<const LANES: usize> = crate::Simd<u8, LANES>;

/// A SIMD vector of containing `LANES` `u16` values.
pub type SimdU16<const LANES: usize> = crate::Simd<u16, LANES>;

/// A SIMD vector of containing `LANES` `u32` values.
pub type SimdU32<const LANES: usize> = crate::Simd<u32, LANES>;

/// A SIMD vector of containing `LANES` `u64` values.
pub type SimdU64<const LANES: usize> = crate::Simd<u64, LANES>;

/// A SIMD vector of containing `LANES` `usize` values.
pub type SimdUsize<const LANES: usize> = crate::Simd<usize, LANES>;

/// Vector of two `usize` values
pub type usizex2 = SimdUsize<2>;

/// Vector of four `usize` values
pub type usizex4 = SimdUsize<4>;

/// Vector of eight `usize` values
pub type usizex8 = SimdUsize<8>;

/// Vector of two `u16` values
pub type u16x2 = SimdU16<2>;

/// Vector of four `u16` values
pub type u16x4 = SimdU16<4>;

/// Vector of eight `u16` values
pub type u16x8 = SimdU16<8>;

/// Vector of 16 `u16` values
pub type u16x16 = SimdU16<16>;

/// Vector of 32 `u16` values
pub type u16x32 = SimdU16<32>;

/// Vector of two `u32` values
pub type u32x2 = SimdU32<2>;

/// Vector of four `u32` values
pub type u32x4 = SimdU32<4>;

/// Vector of eight `u32` values
pub type u32x8 = SimdU32<8>;

/// Vector of 16 `u32` values
pub type u32x16 = SimdU32<16>;

/// Vector of two `u64` values
pub type u64x2 = SimdU64<2>;

/// Vector of four `u64` values
pub type u64x4 = SimdU64<4>;

/// Vector of eight `u64` values
pub type u64x8 = SimdU64<8>;

/// Vector of four `u8` values
pub type u8x4 = SimdU8<4>;

/// Vector of eight `u8` values
pub type u8x8 = SimdU8<8>;

/// Vector of 16 `u8` values
pub type u8x16 = SimdU8<16>;

/// Vector of 32 `u8` values
pub type u8x32 = SimdU8<32>;

/// Vector of 64 `u8` values
pub type u8x64 = SimdU8<64>;

#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `u8` values.
#[repr(simd)]
pub struct SimdU8<const LANES: usize>([u8; LANES]);

impl_integer_vector! { SimdU8, u8 }

/// Vector of eight `u8` values
pub type u8x8 = SimdU8<8>;

/// Vector of 16 `u8` values
pub type u8x16 = SimdU8<16>;

/// Vector of 32 `u8` values
pub type u8x32 = SimdU8<32>;

/// Vector of 64 `u8` values
pub type u8x64 = SimdU8<64>;

from_transmute_x86! { unsafe u8x16 => __m128i }
from_transmute_x86! { unsafe u8x32 => __m256i }
//from_transmute_x86! { unsafe u8x64 => __m512i }

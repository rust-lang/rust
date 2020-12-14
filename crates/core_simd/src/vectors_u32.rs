#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `u32` values.
#[repr(simd)]
pub struct SimdU32<const LANES: usize>([u32; LANES]);

impl_integer_vector! { SimdU32, u32 }

/// Vector of two `u32` values
pub type u32x2 = SimdU32<2>;

/// Vector of four `u32` values
pub type u32x4 = SimdU32<4>;

/// Vector of eight `u32` values
pub type u32x8 = SimdU32<8>;

/// Vector of 16 `u32` values
pub type u32x16 = SimdU32<16>;

from_transmute_x86! { unsafe u32x4 => __m128i }
from_transmute_x86! { unsafe u32x8 => __m256i }
//from_transmute_x86! { unsafe u32x16 => __m512i }

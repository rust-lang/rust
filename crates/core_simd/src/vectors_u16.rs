#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `u16` values.
#[repr(simd)]
pub struct SimdU16<const LANES: usize>([u16; LANES]);

impl_integer_vector! { SimdU16, u16 }

/// Vector of four `u16` values
pub type u16x4 = SimdU16<4>;

/// Vector of eight `u16` values
pub type u16x8 = SimdU16<8>;

/// Vector of 16 `u16` values
pub type u16x16 = SimdU16<16>;

/// Vector of 32 `u16` values
pub type u16x32 = SimdU16<32>;

from_transmute_x86! { unsafe u16x8 => __m128i }
from_transmute_x86! { unsafe u16x16 => __m256i }
//from_transmute_x86! { unsafe u16x32 => __m512i }

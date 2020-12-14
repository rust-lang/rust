#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `i16` values.
#[repr(simd)]
pub struct SimdI16<const LANES: usize>([i16; LANES]);

impl_integer_vector! { SimdI16, i16 }

/// Vector of four `i16` values
pub type i16x4 = SimdI16<4>;

/// Vector of eight `i16` values
pub type i16x8 = SimdI16<8>;

/// Vector of 16 `i16` values
pub type i16x16 = SimdI16<16>;

/// Vector of 32 `i16` values
pub type i16x32 = SimdI16<32>;

from_transmute_x86! { unsafe i16x8 => __m128i }
from_transmute_x86! { unsafe i16x16 => __m256i }
//from_transmute_x86! { unsafe i16x32 => __m512i }

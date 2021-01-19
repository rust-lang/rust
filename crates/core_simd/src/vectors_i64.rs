#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `i64` values.
#[repr(simd)]
pub struct SimdI64<const LANES: usize>([i64; LANES]);

impl_integer_vector! { SimdI64, i64 }

/// Vector of two `i64` values
pub type i64x2 = SimdI64<2>;

/// Vector of four `i64` values
pub type i64x4 = SimdI64<4>;

/// Vector of eight `i64` values
pub type i64x8 = SimdI64<8>;

from_transmute_x86! { unsafe i64x2 => __m128i }
from_transmute_x86! { unsafe i64x4 => __m256i }
//from_transmute_x86! { unsafe i64x8 => __m512i }

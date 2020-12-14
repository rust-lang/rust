#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `i128` values.
#[repr(simd)]
pub struct SimdI128<const LANES: usize>([i128; LANES]);

impl_integer_vector! { SimdI128, i128 }

/// Vector of two `i128` values
pub type i128x2 = SimdI128<2>;

/// Vector of four `i128` values
pub type i128x4 = SimdI128<4>;

from_transmute_x86! { unsafe i128x2 => __m256i }
//from_transmute_x86! { unsafe i128x4 => __m512i }

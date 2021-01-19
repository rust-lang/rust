#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `u128` values.
#[repr(simd)]
pub struct SimdU128<const LANES: usize>([u128; LANES]);

impl_integer_vector! { SimdU128, u128 }

/// Vector of two `u128` values
pub type u128x2 = SimdU128<2>;

/// Vector of four `u128` values
pub type u128x4 = SimdU128<4>;

from_transmute_x86! { unsafe u128x2 => __m256i }
//from_transmute_x86! { unsafe u128x4 => __m512i }

#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `f64` values.
#[repr(simd)]
pub struct SimdF64<const LANES: usize>([f64; LANES]);

impl_float_vector! { SimdF64, f64, SimdU64 }

/// Vector of two `f64` values
pub type f64x2 = SimdF64<2>;

/// Vector of four `f64` values
pub type f64x4 = SimdF64<4>;

/// Vector of eight `f64` values
pub type f64x8 = SimdF64<8>;

from_transmute_x86! { unsafe f64x2 => __m128d }
from_transmute_x86! { unsafe f64x4 => __m256d }
//from_transmute_x86! { unsafe f64x8 => __m512d }

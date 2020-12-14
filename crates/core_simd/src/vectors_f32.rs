#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `f32` values.
#[repr(simd)]
pub struct SimdF32<const LANES: usize>([f32; LANES]);

impl_float_vector! { SimdF32, f32, SimdU32 }

/// Vector of two `f32` values
pub type f32x2 = SimdF32<2>;

/// Vector of four `f32` values
pub type f32x4 = SimdF32<4>;

/// Vector of eight `f32` values
pub type f32x8 = SimdF32<8>;

/// Vector of 16 `f32` values
pub type f32x16 = SimdF32<16>;

from_transmute_x86! { unsafe f32x4 => __m128 }
from_transmute_x86! { unsafe f32x8 => __m256 }
//from_transmute_x86! { unsafe f32x16 => __m512 }

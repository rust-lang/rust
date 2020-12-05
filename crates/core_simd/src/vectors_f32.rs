#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` lanes of `f32`.
#[repr(simd)]
pub struct SimdF32<const LANES: usize>([f32; LANES]);

impl_vector! { SimdF32, f32 }

pub type f32x2 = SimdF32<2>;
pub type f32x4 = SimdF32<4>;
pub type f32x8 = SimdF32<8>;
pub type f32x16 = SimdF32<16>;

from_transmute_x86! { unsafe f32x4 => __m128 }
from_transmute_x86! { unsafe f32x8 => __m256 }
//from_transmute_x86! { unsafe f32x16 => __m512 }

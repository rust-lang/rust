#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` lanes of `i32`.
#[repr(simd)]
pub struct SimdI32<const LANES: usize>([i32; LANES]);

impl_integer_vector! { SimdI32, i32 }

pub type i32x2 = SimdI32<2>;
pub type i32x4 = SimdI32<4>;
pub type i32x8 = SimdI32<8>;
pub type i32x16 = SimdI32<16>;

from_transmute_x86! { unsafe i32x4 => __m128i }
from_transmute_x86! { unsafe i32x8 => __m256i }
//from_transmute_x86! { unsafe i32x16 => __m512i }

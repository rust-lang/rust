#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` lanes of `u64`.
#[repr(simd)]
pub struct SimdU64<const LANES: usize>([u64; LANES]);

impl_vector! { SimdU64, u64 }

pub type u64x2 = SimdU64<2>;
pub type u64x4 = SimdU64<4>;
pub type u64x8 = SimdU64<8>;

from_transmute_x86! { unsafe u64x2 => __m128i }
from_transmute_x86! { unsafe u64x4 => __m256i }
//from_transmute_x86! { unsafe u64x8 => __m512i }

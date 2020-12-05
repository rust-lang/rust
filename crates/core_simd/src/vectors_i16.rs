#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` lanes of `i16`.
#[repr(simd)]
pub struct SimdI16<const LANES: usize>([i16; LANES]);

impl_vector! { SimdI16, i16 }

pub type i16x4 = SimdI16<4>;
pub type i16x8 = SimdI16<8>;
pub type i16x16 = SimdI16<16>;
pub type i16x32 = SimdI16<32>;

from_transmute_x86! { unsafe i16x8 => __m128i }
from_transmute_x86! { unsafe i16x16 => __m256i }
//from_transmute_x86! { unsafe i16x32 => __m512i }

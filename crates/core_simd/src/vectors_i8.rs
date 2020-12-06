#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` lanes of `i8`.
#[repr(simd)]
pub struct SimdI8<const LANES: usize>([i8; LANES]);

impl_integer_vector! { SimdI8, i8 }

pub type i8x8 = SimdI8<8>;
pub type i8x16 = SimdI8<16>;
pub type i8x32 = SimdI8<32>;
pub type i8x64 = SimdI8<64>;

from_transmute_x86! { unsafe i8x16 => __m128i }
from_transmute_x86! { unsafe i8x32 => __m256i }
//from_transmute_x86! { unsafe i8x64 => __m512i }

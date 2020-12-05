#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` lanes of `u8`.
#[repr(simd)]
pub struct SimdU8<const LANES: usize>([u8; LANES]);

impl_vector! { SimdU8, u8 }

pub type u8x8 = SimdU8<8>;
pub type u8x16 = SimdU8<16>;
pub type u8x32 = SimdU8<32>;
pub type u8x64 = SimdU8<64>;

from_transmute_x86! { unsafe u8x16 => __m128i }
from_transmute_x86! { unsafe u8x32 => __m256i }
//from_transmute_x86! { unsafe u8x64 => __m512i }

#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `usize` values.
#[repr(simd)]
pub struct SimdUsize<const LANES: usize>([usize; LANES]);

impl_integer_vector! { SimdUsize, usize }

/// Vector of two `usize` values
pub type usizex2 = SimdUsize<2>;

/// Vector of four `usize` values
pub type usizex4 = SimdUsize<4>;

/// Vector of eight `usize` values
pub type usizex8 = SimdUsize<8>;

#[cfg(target_pointer_width = "32")]
from_transmute_x86! { unsafe usizex4 => __m128i }
#[cfg(target_pointer_width = "32")]
from_transmute_x86! { unsafe usizex8 => __m256i }

#[cfg(target_pointer_width = "64")]
from_transmute_x86! { unsafe usizex2 => __m128i }
#[cfg(target_pointer_width = "64")]
from_transmute_x86! { unsafe usizex4 => __m256i }
//#[cfg(target_pointer_width = "64")]
//from_transmute_x86! { unsafe usizex8 => __m512i }

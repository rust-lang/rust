#![allow(non_camel_case_types)]

/// A SIMD vector of containing `LANES` `isize` values.
#[repr(simd)]
pub struct SimdIsize<const LANES: usize>([isize; LANES]);

impl_integer_vector! { SimdIsize, isize }

/// Vector of two `isize` values
pub type isizex2 = SimdIsize<2>;

/// Vector of four `isize` values
pub type isizex4 = SimdIsize<4>;

/// Vector of eight `isize` values
pub type isizex8 = SimdIsize<8>;

#[cfg(target_pointer_width = "32")]
from_transmute_x86! { unsafe isizex4 => __m128i }
#[cfg(target_pointer_width = "32")]
from_transmute_x86! { unsafe isizex8 => __m256i }

#[cfg(target_pointer_width = "64")]
from_transmute_x86! { unsafe isizex2 => __m128i }
#[cfg(target_pointer_width = "64")]
from_transmute_x86! { unsafe isizex4 => __m256i }
//#[cfg(target_pointer_width = "64")]
//from_transmute_x86! { unsafe isizex8 => __m512i }

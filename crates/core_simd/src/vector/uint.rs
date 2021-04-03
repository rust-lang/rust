#![allow(non_camel_case_types)]


/// Implements additional integer traits (Eq, Ord, Hash) on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! impl_unsigned_vector {
    { $name:ident, $type:ty } => {
        impl_vector! { $name, $type }

        impl<const LANES: usize> Eq for $name<LANES> where Self: crate::LanesAtMost32 {}

        impl<const LANES: usize> Ord for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                // TODO use SIMD cmp
                self.to_array().cmp(other.as_ref())
            }
        }

        impl<const LANES: usize> core::hash::Hash for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn hash<H>(&self, state: &mut H)
            where
                H: core::hash::Hasher
            {
                self.as_slice().hash(state)
            }
        }
    }
}

/// A SIMD vector of containing `LANES` `usize` values.
#[repr(simd)]
pub struct SimdUsize<const LANES: usize>([usize; LANES])
where
    Self: crate::LanesAtMost32;

impl_unsigned_vector! { SimdUsize, usize }

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

/// A SIMD vector of containing `LANES` `u128` values.
#[repr(simd)]
pub struct SimdU128<const LANES: usize>([u128; LANES])
where
    Self: crate::LanesAtMost32;

impl_unsigned_vector! { SimdU128, u128 }

from_transmute_x86! { unsafe u128x2 => __m256i }
//from_transmute_x86! { unsafe u128x4 => __m512i }

/// A SIMD vector of containing `LANES` `u16` values.
#[repr(simd)]
pub struct SimdU16<const LANES: usize>([u16; LANES])
where
    Self: crate::LanesAtMost32;

impl_unsigned_vector! { SimdU16, u16 }

from_transmute_x86! { unsafe u16x8 => __m128i }
from_transmute_x86! { unsafe u16x16 => __m256i }
//from_transmute_x86! { unsafe u16x32 => __m512i }

/// A SIMD vector of containing `LANES` `u32` values.
#[repr(simd)]
pub struct SimdU32<const LANES: usize>([u32; LANES])
where
    Self: crate::LanesAtMost32;

impl_unsigned_vector! { SimdU32, u32 }

from_transmute_x86! { unsafe u32x4 => __m128i }
from_transmute_x86! { unsafe u32x8 => __m256i }
//from_transmute_x86! { unsafe u32x16 => __m512i }

/// A SIMD vector of containing `LANES` `u64` values.
#[repr(simd)]
pub struct SimdU64<const LANES: usize>([u64; LANES])
where
    Self: crate::LanesAtMost32;

impl_unsigned_vector! { SimdU64, u64 }

from_transmute_x86! { unsafe u64x2 => __m128i }
from_transmute_x86! { unsafe u64x4 => __m256i }
//from_transmute_x86! { unsafe u64x8 => __m512i }

/// A SIMD vector of containing `LANES` `u8` values.
#[repr(simd)]
pub struct SimdU8<const LANES: usize>([u8; LANES])
where
    Self: crate::LanesAtMost32;

impl_unsigned_vector! { SimdU8, u8 }

from_transmute_x86! { unsafe u8x16 => __m128i }
from_transmute_x86! { unsafe u8x32 => __m256i }
//from_transmute_x86! { unsafe u8x64 => __m512i }

/// Vector of two `usize` values
pub type usizex2 = SimdUsize<2>;

/// Vector of four `usize` values
pub type usizex4 = SimdUsize<4>;

/// Vector of eight `usize` values
pub type usizex8 = SimdUsize<8>;

/// Vector of two `u128` values
pub type u128x2 = SimdU128<2>;

/// Vector of four `u128` values
pub type u128x4 = SimdU128<4>;

/// Vector of four `u16` values
pub type u16x4 = SimdU16<4>;

/// Vector of eight `u16` values
pub type u16x8 = SimdU16<8>;

/// Vector of 16 `u16` values
pub type u16x16 = SimdU16<16>;

/// Vector of 32 `u16` values
pub type u16x32 = SimdU16<32>;

/// Vector of two `u32` values
pub type u32x2 = SimdU32<2>;

/// Vector of four `u32` values
pub type u32x4 = SimdU32<4>;

/// Vector of eight `u32` values
pub type u32x8 = SimdU32<8>;

/// Vector of 16 `u32` values
pub type u32x16 = SimdU32<16>;

/// Vector of two `u64` values
pub type u64x2 = SimdU64<2>;

/// Vector of four `u64` values
pub type u64x4 = SimdU64<4>;

/// Vector of eight `u64` values
pub type u64x8 = SimdU64<8>;

/// Vector of eight `u8` values
pub type u8x8 = SimdU8<8>;

/// Vector of 16 `u8` values
pub type u8x16 = SimdU8<16>;

/// Vector of 32 `u8` values
pub type u8x32 = SimdU8<32>;

/// Vector of 64 `u8` values
pub type u8x64 = SimdU8<64>;

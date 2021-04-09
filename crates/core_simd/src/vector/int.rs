#![allow(non_camel_case_types)]

/// Implements additional integer traits (Eq, Ord, Hash) on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! impl_integer_vector {
    { $name:ident, $type:ty, $mask_ty:ident, $mask_impl_ty:ident } => {
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

        impl<const LANES: usize> $name<LANES>
        where
            Self: crate::LanesAtMost32,
            crate::$mask_impl_ty<LANES>: crate::LanesAtMost32,
        {
            /// Returns true for each positive lane and false if it is zero or negative.
            pub fn is_positive(self) -> crate::$mask_ty<LANES> {
                self.lanes_gt(Self::splat(0))
            }

            /// Returns true for each negative lane and false if it is zero or positive.
            pub fn is_negative(self) -> crate::$mask_ty<LANES> {
                self.lanes_lt(Self::splat(0))
            }
        }
    }
}

/// A SIMD vector of containing `LANES` `isize` values.
#[repr(simd)]
pub struct SimdIsize<const LANES: usize>([isize; LANES])
where
    Self: crate::LanesAtMost32;

impl_integer_vector! { SimdIsize, isize, MaskSize, SimdIsize }

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

/// A SIMD vector of containing `LANES` `i128` values.
#[repr(simd)]
pub struct SimdI128<const LANES: usize>([i128; LANES])
where
    Self: crate::LanesAtMost32;

impl_integer_vector! { SimdI128, i128, Mask128, SimdI128 }

from_transmute_x86! { unsafe i128x2 => __m256i }
//from_transmute_x86! { unsafe i128x4 => __m512i }

/// A SIMD vector of containing `LANES` `i16` values.
#[repr(simd)]
pub struct SimdI16<const LANES: usize>([i16; LANES])
where
    Self: crate::LanesAtMost32;

impl_integer_vector! { SimdI16, i16, Mask16, SimdI16 }

from_transmute_x86! { unsafe i16x8 => __m128i }
from_transmute_x86! { unsafe i16x16 => __m256i }
//from_transmute_x86! { unsafe i16x32 => __m512i }

/// A SIMD vector of containing `LANES` `i32` values.
#[repr(simd)]
pub struct SimdI32<const LANES: usize>([i32; LANES])
where
    Self: crate::LanesAtMost32;

impl_integer_vector! { SimdI32, i32, Mask32, SimdI32 }

from_transmute_x86! { unsafe i32x4 => __m128i }
from_transmute_x86! { unsafe i32x8 => __m256i }
//from_transmute_x86! { unsafe i32x16 => __m512i }

/// A SIMD vector of containing `LANES` `i64` values.
#[repr(simd)]
pub struct SimdI64<const LANES: usize>([i64; LANES])
where
    Self: crate::LanesAtMost32;

impl_integer_vector! { SimdI64, i64, Mask64, SimdI64 }

from_transmute_x86! { unsafe i64x2 => __m128i }
from_transmute_x86! { unsafe i64x4 => __m256i }
//from_transmute_x86! { unsafe i64x8 => __m512i }

/// A SIMD vector of containing `LANES` `i8` values.
#[repr(simd)]
pub struct SimdI8<const LANES: usize>([i8; LANES])
where
    Self: crate::LanesAtMost32;

impl_integer_vector! { SimdI8, i8, Mask8, SimdI8 }

from_transmute_x86! { unsafe i8x16 => __m128i }
from_transmute_x86! { unsafe i8x32 => __m256i }
//from_transmute_x86! { unsafe i8x64 => __m512i }

/// Vector of two `isize` values
pub type isizex2 = SimdIsize<2>;

/// Vector of four `isize` values
pub type isizex4 = SimdIsize<4>;

/// Vector of eight `isize` values
pub type isizex8 = SimdIsize<8>;

/// Vector of two `i128` values
pub type i128x2 = SimdI128<2>;

/// Vector of four `i128` values
pub type i128x4 = SimdI128<4>;

/// Vector of four `i16` values
pub type i16x4 = SimdI16<4>;

/// Vector of eight `i16` values
pub type i16x8 = SimdI16<8>;

/// Vector of 16 `i16` values
pub type i16x16 = SimdI16<16>;

/// Vector of 32 `i16` values
pub type i16x32 = SimdI16<32>;

/// Vector of two `i32` values
pub type i32x2 = SimdI32<2>;

/// Vector of four `i32` values
pub type i32x4 = SimdI32<4>;

/// Vector of eight `i32` values
pub type i32x8 = SimdI32<8>;

/// Vector of 16 `i32` values
pub type i32x16 = SimdI32<16>;

/// Vector of two `i64` values
pub type i64x2 = SimdI64<2>;

/// Vector of four `i64` values
pub type i64x4 = SimdI64<4>;

/// Vector of eight `i64` values
pub type i64x8 = SimdI64<8>;

/// Vector of eight `i8` values
pub type i8x8 = SimdI8<8>;

/// Vector of 16 `i8` values
pub type i8x16 = SimdI8<16>;

/// Vector of 32 `i8` values
pub type i8x32 = SimdI8<32>;

/// Vector of 64 `i8` values
pub type i8x64 = SimdI8<64>;

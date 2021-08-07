#![allow(non_camel_case_types)]

use crate::{LaneCount, SupportedLaneCount};

/// Implements additional integer traits (Eq, Ord, Hash) on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! impl_integer_vector {
    { $name:ident, $type:ty, $mask_ty:ident, $mask_impl_ty:ident } => {
        impl<const LANES: usize> $name<LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            /// Returns true for each positive lane and false if it is zero or negative.
            #[inline]
            pub fn is_positive(self) -> crate::$mask_ty<LANES> {
                self.lanes_gt(Self::splat(0))
            }

            /// Returns true for each negative lane and false if it is zero or positive.
            #[inline]
            pub fn is_negative(self) -> crate::$mask_ty<LANES> {
                self.lanes_lt(Self::splat(0))
            }

            /// Returns numbers representing the sign of each lane.
            /// * `0` if the number is zero
            /// * `1` if the number is positive
            /// * `-1` if the number is negative
            #[inline]
            pub fn signum(self) -> Self {
                self.is_positive().select(
                    Self::splat(1),
                    self.is_negative().select(Self::splat(-1), Self::splat(0))
                )
            }
        }
    }
}

/// A SIMD vector of containing `LANES` `i8` values.
pub type SimdI8<const LANES: usize> = crate::Simd<i8, LANES>;

/// A SIMD vector of containing `LANES` `i16` values.
pub type SimdI16<const LANES: usize> = crate::Simd<i16, LANES>;

/// A SIMD vector of containing `LANES` `i32` values.
pub type SimdI32<const LANES: usize> = crate::Simd<i32, LANES>;

/// A SIMD vector of containing `LANES` `i64` values.
pub type SimdI64<const LANES: usize> = crate::Simd<i64, LANES>;

/// A SIMD vector of containing `LANES` `isize` values.
pub type SimdIsize<const LANES: usize> = crate::Simd<isize, LANES>;

impl_integer_vector! { SimdIsize, isize, MaskSize, SimdIsize }
impl_integer_vector! { SimdI16, i16, Mask16, SimdI16 }
impl_integer_vector! { SimdI32, i32, Mask32, SimdI32 }
impl_integer_vector! { SimdI64, i64, Mask64, SimdI64 }
impl_integer_vector! { SimdI8, i8, Mask8, SimdI8 }

/// Vector of two `isize` values
pub type isizex2 = SimdIsize<2>;

/// Vector of four `isize` values
pub type isizex4 = SimdIsize<4>;

/// Vector of eight `isize` values
pub type isizex8 = SimdIsize<8>;

/// Vector of two `i16` values
pub type i16x2 = SimdI16<2>;

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

/// Vector of four `i8` values
pub type i8x4 = SimdI8<4>;

/// Vector of eight `i8` values
pub type i8x8 = SimdI8<8>;

/// Vector of 16 `i8` values
pub type i8x16 = SimdI8<16>;

/// Vector of 32 `i8` values
pub type i8x32 = SimdI8<32>;

/// Vector of 64 `i8` values
pub type i8x64 = SimdI8<64>;

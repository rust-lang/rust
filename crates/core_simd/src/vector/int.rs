#![allow(non_camel_case_types)]

use crate::simd::{LaneCount, Mask, Simd, SupportedLaneCount};

/// Implements additional integer traits (Eq, Ord, Hash) on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! impl_integer_vector {
    { $type:ty } => {
        impl<const LANES: usize> Simd<$type, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            /// Returns true for each positive lane and false if it is zero or negative.
            #[inline]
            pub fn is_positive(self) -> Mask<$type, LANES> {
                self.lanes_gt(Self::splat(0))
            }

            /// Returns true for each negative lane and false if it is zero or positive.
            #[inline]
            pub fn is_negative(self) -> Mask<$type, LANES> {
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

impl_integer_vector! { isize }
impl_integer_vector! { i16 }
impl_integer_vector! { i32 }
impl_integer_vector! { i64 }
impl_integer_vector! { i8 }

/// A SIMD vector with two elements of type `isize`.
pub type isizex2 = Simd<isize, 2>;

/// A SIMD vector with four elements of type `isize`.
pub type isizex4 = Simd<isize, 4>;

/// A SIMD vector with eight elements of type `isize`.
pub type isizex8 = Simd<isize, 8>;

/// A 32-bit SIMD vector with two elements of type `i16`.
pub type i16x2 = Simd<i16, 2>;

/// A 64-bit SIMD vector with four elements of type `i16`.
pub type i16x4 = Simd<i16, 4>;

/// A 128-bit SIMD vector with eight elements of type `i16`.
pub type i16x8 = Simd<i16, 8>;

/// A 256-bit SIMD vector with 16 elements of type `i16`.
pub type i16x16 = Simd<i16, 16>;

/// A 512-bit SIMD vector with 32 elements of type `i16`.
pub type i16x32 = Simd<i16, 32>;

/// A 64-bit SIMD vector with two elements of type `i32`.
pub type i32x2 = Simd<i32, 2>;

/// A 128-bit SIMD vector with four elements of type `i32`.
pub type i32x4 = Simd<i32, 4>;

/// A 256-bit SIMD vector with eight elements of type `i32`.
pub type i32x8 = Simd<i32, 8>;

/// A 512-bit SIMD vector with 16 elements of type `i32`.
pub type i32x16 = Simd<i32, 16>;

/// A 128-bit SIMD vector with two elements of type `i64`.
pub type i64x2 = Simd<i64, 2>;

/// A 256-bit SIMD vector with four elements of type `i64`.
pub type i64x4 = Simd<i64, 4>;

/// A 512-bit SIMD vector with eight elements of type `i64`.
pub type i64x8 = Simd<i64, 8>;

/// A 32-bit SIMD vector with four elements of type `i8`.
pub type i8x4 = Simd<i8, 4>;

/// A 64-bit SIMD vector with eight elements of type `i8`.
pub type i8x8 = Simd<i8, 8>;

/// A 128-bit SIMD vector with 16 elements of type `i8`.
pub type i8x16 = Simd<i8, 16>;

/// A 256-bit SIMD vector with 32 elements of type `i8`.
pub type i8x32 = Simd<i8, 32>;

/// A 512-bit SIMD vector with 64 elements of type `i8`.
pub type i8x64 = Simd<i8, 64>;

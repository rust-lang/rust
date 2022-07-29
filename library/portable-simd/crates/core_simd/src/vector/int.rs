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

/// Vector of two `isize` values
pub type isizex2 = Simd<isize, 2>;

/// Vector of four `isize` values
pub type isizex4 = Simd<isize, 4>;

/// Vector of eight `isize` values
pub type isizex8 = Simd<isize, 8>;

/// Vector of two `i16` values
pub type i16x2 = Simd<i16, 2>;

/// Vector of four `i16` values
pub type i16x4 = Simd<i16, 4>;

/// Vector of eight `i16` values
pub type i16x8 = Simd<i16, 8>;

/// Vector of 16 `i16` values
pub type i16x16 = Simd<i16, 16>;

/// Vector of 32 `i16` values
pub type i16x32 = Simd<i16, 32>;

/// Vector of two `i32` values
pub type i32x2 = Simd<i32, 2>;

/// Vector of four `i32` values
pub type i32x4 = Simd<i32, 4>;

/// Vector of eight `i32` values
pub type i32x8 = Simd<i32, 8>;

/// Vector of 16 `i32` values
pub type i32x16 = Simd<i32, 16>;

/// Vector of two `i64` values
pub type i64x2 = Simd<i64, 2>;

/// Vector of four `i64` values
pub type i64x4 = Simd<i64, 4>;

/// Vector of eight `i64` values
pub type i64x8 = Simd<i64, 8>;

/// Vector of four `i8` values
pub type i8x4 = Simd<i8, 4>;

/// Vector of eight `i8` values
pub type i8x8 = Simd<i8, 8>;

/// Vector of 16 `i8` values
pub type i8x16 = Simd<i8, 16>;

/// Vector of 32 `i8` values
pub type i8x32 = Simd<i8, 32>;

/// Vector of 64 `i8` values
pub type i8x64 = Simd<i8, 64>;

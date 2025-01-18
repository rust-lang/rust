use super::sealed::Sealed;
use crate::simd::{cmp::SimdOrd, LaneCount, Simd, SimdCast, SimdElement, SupportedLaneCount};

/// Operations on SIMD vectors of unsigned integers.
pub trait SimdUint: Copy + Sealed {
    /// Scalar type contained by this SIMD vector type.
    type Scalar;

    /// A SIMD vector with a different element type.
    type Cast<T: SimdElement>;

    /// Performs elementwise conversion of this vector's elements to another SIMD-valid type.
    ///
    /// This follows the semantics of Rust's `as` conversion for casting integers (wrapping to
    /// other integer types, and saturating to float types).
    #[must_use]
    fn cast<T: SimdCast>(self) -> Self::Cast<T>;

    /// Wrapping negation.
    ///
    /// Like [`u32::wrapping_neg`], all applications of this function will wrap, with the exception
    /// of `-0`.
    fn wrapping_neg(self) -> Self;

    /// Lanewise saturating add.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::u32::MAX;
    /// let x = Simd::from_array([2, 1, 0, MAX]);
    /// let max = Simd::splat(MAX);
    /// let unsat = x + max;
    /// let sat = x.saturating_add(max);
    /// assert_eq!(unsat, Simd::from_array([1, 0, MAX, MAX - 1]));
    /// assert_eq!(sat, max);
    /// ```
    fn saturating_add(self, second: Self) -> Self;

    /// Lanewise saturating subtract.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::u32::MAX;
    /// let x = Simd::from_array([2, 1, 0, MAX]);
    /// let max = Simd::splat(MAX);
    /// let unsat = x - max;
    /// let sat = x.saturating_sub(max);
    /// assert_eq!(unsat, Simd::from_array([3, 2, 1, 0]));
    /// assert_eq!(sat, Simd::splat(0));
    fn saturating_sub(self, second: Self) -> Self;

    /// Lanewise absolute difference.
    /// Every element becomes the absolute difference of `self` and `second`.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::u32::MAX;
    /// let a = Simd::from_array([0, MAX, 100, 20]);
    /// let b = Simd::from_array([MAX, 0, 80, 200]);
    /// assert_eq!(a.abs_diff(b), Simd::from_array([MAX, MAX, 20, 180]));
    /// ```
    fn abs_diff(self, second: Self) -> Self;

    /// Returns the sum of the elements of the vector, with wrapping addition.
    fn reduce_sum(self) -> Self::Scalar;

    /// Returns the product of the elements of the vector, with wrapping multiplication.
    fn reduce_product(self) -> Self::Scalar;

    /// Returns the maximum element in the vector.
    fn reduce_max(self) -> Self::Scalar;

    /// Returns the minimum element in the vector.
    fn reduce_min(self) -> Self::Scalar;

    /// Returns the cumulative bitwise "and" across the elements of the vector.
    fn reduce_and(self) -> Self::Scalar;

    /// Returns the cumulative bitwise "or" across the elements of the vector.
    fn reduce_or(self) -> Self::Scalar;

    /// Returns the cumulative bitwise "xor" across the elements of the vector.
    fn reduce_xor(self) -> Self::Scalar;

    /// Reverses the byte order of each element.
    fn swap_bytes(self) -> Self;

    /// Reverses the order of bits in each elemnent.
    /// The least significant bit becomes the most significant bit, second least-significant bit becomes second most-significant bit, etc.
    fn reverse_bits(self) -> Self;

    /// Returns the number of ones in the binary representation of each element.
    fn count_ones(self) -> Self;

    /// Returns the number of zeros in the binary representation of each element.
    fn count_zeros(self) -> Self;

    /// Returns the number of leading zeros in the binary representation of each element.
    fn leading_zeros(self) -> Self;

    /// Returns the number of trailing zeros in the binary representation of each element.
    fn trailing_zeros(self) -> Self;

    /// Returns the number of leading ones in the binary representation of each element.
    fn leading_ones(self) -> Self;

    /// Returns the number of trailing ones in the binary representation of each element.
    fn trailing_ones(self) -> Self;
}

macro_rules! impl_trait {
    { $($ty:ident ($signed:ident)),* } => {
        $(
        impl<const N: usize> Sealed for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
        }

        impl<const N: usize> SimdUint for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Scalar = $ty;
            type Cast<T: SimdElement> = Simd<T, N>;

            #[inline]
            fn cast<T: SimdCast>(self) -> Self::Cast<T> {
                // Safety: supported types are guaranteed by SimdCast
                unsafe { core::intrinsics::simd::simd_as(self) }
            }

            #[inline]
            fn wrapping_neg(self) -> Self {
                use crate::simd::num::SimdInt;
                (-self.cast::<$signed>()).cast()
            }

            #[inline]
            fn saturating_add(self, second: Self) -> Self {
                // Safety: `self` is a vector
                unsafe { core::intrinsics::simd::simd_saturating_add(self, second) }
            }

            #[inline]
            fn saturating_sub(self, second: Self) -> Self {
                // Safety: `self` is a vector
                unsafe { core::intrinsics::simd::simd_saturating_sub(self, second) }
            }

            #[inline]
            fn abs_diff(self, second: Self) -> Self {
                let max = self.simd_max(second);
                let min = self.simd_min(second);
                max - min
            }

            #[inline]
            fn reduce_sum(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_add_ordered(self, 0) }
            }

            #[inline]
            fn reduce_product(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_mul_ordered(self, 1) }
            }

            #[inline]
            fn reduce_max(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_max(self) }
            }

            #[inline]
            fn reduce_min(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_min(self) }
            }

            #[inline]
            fn reduce_and(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_and(self) }
            }

            #[inline]
            fn reduce_or(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_or(self) }
            }

            #[inline]
            fn reduce_xor(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_xor(self) }
            }

            #[inline]
            fn swap_bytes(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_bswap(self) }
            }

            #[inline]
            fn reverse_bits(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_bitreverse(self) }
            }

            #[inline]
            fn count_ones(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_ctpop(self) }
            }

            #[inline]
            fn count_zeros(self) -> Self {
                (!self).count_ones()
            }

            #[inline]
            fn leading_zeros(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_ctlz(self) }
            }

            #[inline]
            fn trailing_zeros(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_cttz(self) }
            }

            #[inline]
            fn leading_ones(self) -> Self {
                (!self).leading_zeros()
            }

            #[inline]
            fn trailing_ones(self) -> Self {
                (!self).trailing_zeros()
            }
        }
        )*
    }
}

impl_trait! { u8 (i8), u16 (i16), u32 (i32), u64 (i64), usize (isize) }

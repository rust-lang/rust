use super::sealed::Sealed;
use crate::simd::{
    cmp::SimdOrd, cmp::SimdPartialOrd, num::SimdUint, LaneCount, Mask, Simd, SimdCast, SimdElement,
    SupportedLaneCount,
};

/// Operations on SIMD vectors of signed integers.
pub trait SimdInt: Copy + Sealed {
    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Scalar type contained by this SIMD vector type.
    type Scalar;

    /// A SIMD vector of unsigned integers with the same element size.
    type Unsigned;

    /// A SIMD vector with a different element type.
    type Cast<T: SimdElement>;

    /// Performs elementwise conversion of this vector's elements to another SIMD-valid type.
    ///
    /// This follows the semantics of Rust's `as` conversion for casting integers (wrapping to
    /// other integer types, and saturating to float types).
    #[must_use]
    fn cast<T: SimdCast>(self) -> Self::Cast<T>;

    /// Lanewise saturating add.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::i32::{MIN, MAX};
    /// let x = Simd::from_array([MIN, 0, 1, MAX]);
    /// let max = Simd::splat(MAX);
    /// let unsat = x + max;
    /// let sat = x.saturating_add(max);
    /// assert_eq!(unsat, Simd::from_array([-1, MAX, MIN, -2]));
    /// assert_eq!(sat, Simd::from_array([-1, MAX, MAX, MAX]));
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
    /// use core::i32::{MIN, MAX};
    /// let x = Simd::from_array([MIN, -2, -1, MAX]);
    /// let max = Simd::splat(MAX);
    /// let unsat = x - max;
    /// let sat = x.saturating_sub(max);
    /// assert_eq!(unsat, Simd::from_array([1, MAX, MIN, 0]));
    /// assert_eq!(sat, Simd::from_array([MIN, MIN, MIN, 0]));
    fn saturating_sub(self, second: Self) -> Self;

    /// Lanewise absolute value, implemented in Rust.
    /// Every element becomes its absolute value.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::i32::{MIN, MAX};
    /// let xs = Simd::from_array([MIN, MIN + 1, -5, 0]);
    /// assert_eq!(xs.abs(), Simd::from_array([MIN, MAX, 5, 0]));
    /// ```
    fn abs(self) -> Self;

    /// Lanewise absolute difference.
    /// Every element becomes the absolute difference of `self` and `second`.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::i32::{MIN, MAX};
    /// let a = Simd::from_array([MIN, MAX, 100, -100]);
    /// let b = Simd::from_array([MAX, MIN, -80, -120]);
    /// assert_eq!(a.abs_diff(b), Simd::from_array([u32::MAX, u32::MAX, 180, 20]));
    /// ```
    fn abs_diff(self, second: Self) -> Self::Unsigned;

    /// Lanewise saturating absolute value, implemented in Rust.
    /// As abs(), except the MIN value becomes MAX instead of itself.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::i32::{MIN, MAX};
    /// let xs = Simd::from_array([MIN, -2, 0, 3]);
    /// let unsat = xs.abs();
    /// let sat = xs.saturating_abs();
    /// assert_eq!(unsat, Simd::from_array([MIN, 2, 0, 3]));
    /// assert_eq!(sat, Simd::from_array([MAX, 2, 0, 3]));
    /// ```
    fn saturating_abs(self) -> Self;

    /// Lanewise saturating negation, implemented in Rust.
    /// As neg(), except the MIN value becomes MAX instead of itself.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// use core::i32::{MIN, MAX};
    /// let x = Simd::from_array([MIN, -2, 3, MAX]);
    /// let unsat = -x;
    /// let sat = x.saturating_neg();
    /// assert_eq!(unsat, Simd::from_array([MIN, 2, -3, MIN + 1]));
    /// assert_eq!(sat, Simd::from_array([MAX, 2, -3, MIN + 1]));
    /// ```
    fn saturating_neg(self) -> Self;

    /// Returns true for each positive element and false if it is zero or negative.
    fn is_positive(self) -> Self::Mask;

    /// Returns true for each negative element and false if it is zero or positive.
    fn is_negative(self) -> Self::Mask;

    /// Returns numbers representing the sign of each element.
    /// * `0` if the number is zero
    /// * `1` if the number is positive
    /// * `-1` if the number is negative
    fn signum(self) -> Self;

    /// Returns the sum of the elements of the vector, with wrapping addition.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_sum(), 10);
    ///
    /// // SIMD integer addition is always wrapping
    /// let v = i32x4::from_array([i32::MAX, 1, 0, 0]);
    /// assert_eq!(v.reduce_sum(), i32::MIN);
    /// ```
    fn reduce_sum(self) -> Self::Scalar;

    /// Returns the product of the elements of the vector, with wrapping multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_product(), 24);
    ///
    /// // SIMD integer multiplication is always wrapping
    /// let v = i32x4::from_array([i32::MAX, 2, 1, 1]);
    /// assert!(v.reduce_product() < i32::MAX);
    /// ```
    fn reduce_product(self) -> Self::Scalar;

    /// Returns the maximum element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_max(), 4);
    /// ```
    fn reduce_max(self) -> Self::Scalar;

    /// Returns the minimum element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_min(), 1);
    /// ```
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
    fn count_ones(self) -> Self::Unsigned;

    /// Returns the number of zeros in the binary representation of each element.
    fn count_zeros(self) -> Self::Unsigned;

    /// Returns the number of leading zeros in the binary representation of each element.
    fn leading_zeros(self) -> Self::Unsigned;

    /// Returns the number of trailing zeros in the binary representation of each element.
    fn trailing_zeros(self) -> Self::Unsigned;

    /// Returns the number of leading ones in the binary representation of each element.
    fn leading_ones(self) -> Self::Unsigned;

    /// Returns the number of trailing ones in the binary representation of each element.
    fn trailing_ones(self) -> Self::Unsigned;
}

macro_rules! impl_trait {
    { $($ty:ident ($unsigned:ident)),* } => {
        $(
        impl<const N: usize> Sealed for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
        }

        impl<const N: usize> SimdInt for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Mask = Mask<<$ty as SimdElement>::Mask, N>;
            type Scalar = $ty;
            type Unsigned = Simd<$unsigned, N>;
            type Cast<T: SimdElement> = Simd<T, N>;

            #[inline]
            fn cast<T: SimdCast>(self) -> Self::Cast<T> {
                // Safety: supported types are guaranteed by SimdCast
                unsafe { core::intrinsics::simd::simd_as(self) }
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
            fn abs(self) -> Self {
                const SHR: $ty = <$ty>::BITS as $ty - 1;
                let m = self >> Simd::splat(SHR);
                (self^m) - m
            }

            #[inline]
            fn abs_diff(self, second: Self) -> Self::Unsigned {
                let max = self.simd_max(second);
                let min = self.simd_min(second);
                (max - min).cast()
            }

            #[inline]
            fn saturating_abs(self) -> Self {
                // arith shift for -1 or 0 mask based on sign bit, giving 2s complement
                const SHR: $ty = <$ty>::BITS as $ty - 1;
                let m = self >> Simd::splat(SHR);
                (self^m).saturating_sub(m)
            }

            #[inline]
            fn saturating_neg(self) -> Self {
                Self::splat(0).saturating_sub(self)
            }

            #[inline]
            fn is_positive(self) -> Self::Mask {
                self.simd_gt(Self::splat(0))
            }

            #[inline]
            fn is_negative(self) -> Self::Mask {
                self.simd_lt(Self::splat(0))
            }

            #[inline]
            fn signum(self) -> Self {
                self.is_positive().select(
                    Self::splat(1),
                    self.is_negative().select(Self::splat(-1), Self::splat(0))
                )
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
            fn count_ones(self) -> Self::Unsigned {
                self.cast::<$unsigned>().count_ones()
            }

            #[inline]
            fn count_zeros(self) -> Self::Unsigned {
                self.cast::<$unsigned>().count_zeros()
            }

            #[inline]
            fn leading_zeros(self) -> Self::Unsigned {
                self.cast::<$unsigned>().leading_zeros()
            }

            #[inline]
            fn trailing_zeros(self) -> Self::Unsigned {
                self.cast::<$unsigned>().trailing_zeros()
            }

            #[inline]
            fn leading_ones(self) -> Self::Unsigned {
                self.cast::<$unsigned>().leading_ones()
            }

            #[inline]
            fn trailing_ones(self) -> Self::Unsigned {
                self.cast::<$unsigned>().trailing_ones()
            }
        }
        )*
    }
}

impl_trait! { i8 (u8), i16 (u16), i32 (u32), i64 (u64), isize (usize) }

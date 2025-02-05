use super::sealed::Sealed;
use crate::simd::{
    cmp::{SimdPartialEq, SimdPartialOrd},
    LaneCount, Mask, Simd, SimdCast, SimdElement, SupportedLaneCount,
};

/// Operations on SIMD vectors of floats.
pub trait SimdFloat: Copy + Sealed {
    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Scalar type contained by this SIMD vector type.
    type Scalar;

    /// Bit representation of this SIMD vector type.
    type Bits;

    /// A SIMD vector with a different element type.
    type Cast<T: SimdElement>;

    /// Performs elementwise conversion of this vector's elements to another SIMD-valid type.
    ///
    /// This follows the semantics of Rust's `as` conversion for floats (truncating or saturating
    /// at the limits) for each element.
    ///
    /// # Example
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let floats: Simd<f32, 4> = Simd::from_array([1.9, -4.5, f32::INFINITY, f32::NAN]);
    /// let ints = floats.cast::<i32>();
    /// assert_eq!(ints, Simd::from_array([1, -4, i32::MAX, 0]));
    ///
    /// // Formally equivalent, but `Simd::cast` can optimize better.
    /// assert_eq!(ints, Simd::from_array(floats.to_array().map(|x| x as i32)));
    ///
    /// // The float conversion does not round-trip.
    /// let floats_again = ints.cast();
    /// assert_ne!(floats, floats_again);
    /// assert_eq!(floats_again, Simd::from_array([1.0, -4.0, 2147483647.0, 0.0]));
    /// ```
    #[must_use]
    fn cast<T: SimdCast>(self) -> Self::Cast<T>;

    /// Rounds toward zero and converts to the same-width integer type, assuming that
    /// the value is finite and fits in that type.
    ///
    /// # Safety
    /// The value must:
    ///
    /// * Not be NaN
    /// * Not be infinite
    /// * Be representable in the return type, after truncating off its fractional part
    ///
    /// If these requirements are infeasible or costly, consider using the safe function [cast],
    /// which saturates on conversion.
    ///
    /// [cast]: Simd::cast
    unsafe fn to_int_unchecked<I: SimdCast>(self) -> Self::Cast<I>
    where
        Self::Scalar: core::convert::FloatToInt<I>;

    /// Raw transmutation to an unsigned integer vector type with the
    /// same size and number of elements.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn to_bits(self) -> Self::Bits;

    /// Raw transmutation from an unsigned integer vector type with the
    /// same size and number of elements.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn from_bits(bits: Self::Bits) -> Self;

    /// Produces a vector where every element has the absolute value of the
    /// equivalently-indexed element in `self`.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn abs(self) -> Self;

    /// Takes the reciprocal (inverse) of each element, `1/x`.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn recip(self) -> Self;

    /// Converts each element from radians to degrees.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn to_degrees(self) -> Self;

    /// Converts each element from degrees to radians.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn to_radians(self) -> Self;

    /// Returns true for each element if it has a positive sign, including
    /// `+0.0`, `NaN`s with positive sign bit and positive infinity.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_sign_positive(self) -> Self::Mask;

    /// Returns true for each element if it has a negative sign, including
    /// `-0.0`, `NaN`s with negative sign bit and negative infinity.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_sign_negative(self) -> Self::Mask;

    /// Returns true for each element if its value is `NaN`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_nan(self) -> Self::Mask;

    /// Returns true for each element if its value is positive infinity or negative infinity.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_infinite(self) -> Self::Mask;

    /// Returns true for each element if its value is neither infinite nor `NaN`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_finite(self) -> Self::Mask;

    /// Returns true for each element if its value is subnormal.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_subnormal(self) -> Self::Mask;

    /// Returns true for each element if its value is neither zero, infinite,
    /// subnormal, nor `NaN`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn is_normal(self) -> Self::Mask;

    /// Replaces each element with a number that represents its sign.
    ///
    /// * `1.0` if the number is positive, `+0.0`, or `INFINITY`
    /// * `-1.0` if the number is negative, `-0.0`, or `NEG_INFINITY`
    /// * `NAN` if the number is `NAN`
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn signum(self) -> Self;

    /// Returns each element with the magnitude of `self` and the sign of `sign`.
    ///
    /// For any element containing a `NAN`, a `NAN` with the sign of `sign` is returned.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn copysign(self, sign: Self) -> Self;

    /// Returns the minimum of each element.
    ///
    /// If one of the values is `NAN`, then the other value is returned.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn simd_min(self, other: Self) -> Self;

    /// Returns the maximum of each element.
    ///
    /// If one of the values is `NAN`, then the other value is returned.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn simd_max(self, other: Self) -> Self;

    /// Restrict each element to a certain interval unless it is NaN.
    ///
    /// For each element in `self`, returns the corresponding element in `max` if the element is
    /// greater than `max`, and the corresponding element in `min` if the element is less
    /// than `min`.  Otherwise returns the element in `self`.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn simd_clamp(self, min: Self, max: Self) -> Self;

    /// Returns the sum of the elements of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = f32x2::from_array([1., 2.]);
    /// assert_eq!(v.reduce_sum(), 3.);
    /// ```
    fn reduce_sum(self) -> Self::Scalar;

    /// Reducing multiply.  Returns the product of the elements of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = f32x2::from_array([3., 4.]);
    /// assert_eq!(v.reduce_product(), 12.);
    /// ```
    fn reduce_product(self) -> Self::Scalar;

    /// Returns the maximum element in the vector.
    ///
    /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
    /// return either.
    ///
    /// This function will not return `NaN` unless all elements are `NaN`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = f32x2::from_array([1., 2.]);
    /// assert_eq!(v.reduce_max(), 2.);
    ///
    /// // NaN values are skipped...
    /// let v = f32x2::from_array([1., f32::NAN]);
    /// assert_eq!(v.reduce_max(), 1.);
    ///
    /// // ...unless all values are NaN
    /// let v = f32x2::from_array([f32::NAN, f32::NAN]);
    /// assert!(v.reduce_max().is_nan());
    /// ```
    fn reduce_max(self) -> Self::Scalar;

    /// Returns the minimum element in the vector.
    ///
    /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
    /// return either.
    ///
    /// This function will not return `NaN` unless all elements are `NaN`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = f32x2::from_array([3., 7.]);
    /// assert_eq!(v.reduce_min(), 3.);
    ///
    /// // NaN values are skipped...
    /// let v = f32x2::from_array([1., f32::NAN]);
    /// assert_eq!(v.reduce_min(), 1.);
    ///
    /// // ...unless all values are NaN
    /// let v = f32x2::from_array([f32::NAN, f32::NAN]);
    /// assert!(v.reduce_min().is_nan());
    /// ```
    fn reduce_min(self) -> Self::Scalar;
}

macro_rules! impl_trait {
    { $($ty:ty { bits: $bits_ty:ty, mask: $mask_ty:ty }),* } => {
        $(
        impl<const N: usize> Sealed for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
        }

        impl<const N: usize> SimdFloat for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Mask = Mask<<$mask_ty as SimdElement>::Mask, N>;
            type Scalar = $ty;
            type Bits = Simd<$bits_ty, N>;
            type Cast<T: SimdElement> = Simd<T, N>;

            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            fn cast<T: SimdCast>(self) -> Self::Cast<T>
            {
                // Safety: supported types are guaranteed by SimdCast
                unsafe { core::intrinsics::simd::simd_as(self) }
            }

            // https://github.com/llvm/llvm-project/issues/94694
            #[cfg(target_arch = "aarch64")]
            #[inline]
            fn cast<T: SimdCast>(self) -> Self::Cast<T>
            {
                const { assert!(N <= 64) };
                if N <= 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64 {
                    // Safety: supported types are guaranteed by SimdCast
                    unsafe { core::intrinsics::simd::simd_as(self) }
                } else if N < 4 {
                    let x = self.resize::<4>(Default::default()).cast();
                    x.resize::<N>(x[0])
                } else if N < 8 {
                    let x = self.resize::<8>(Default::default()).cast();
                    x.resize::<N>(x[0])
                } else if N < 16 {
                    let x = self.resize::<16>(Default::default()).cast();
                    x.resize::<N>(x[0])
                } else if N < 32 {
                    let x = self.resize::<32>(Default::default()).cast();
                    x.resize::<N>(x[0])
                } else {
                    let x = self.resize::<64>(Default::default()).cast();
                    x.resize::<N>(x[0])
                }
            }

            #[inline]
            #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
            unsafe fn to_int_unchecked<I: SimdCast>(self) -> Self::Cast<I>
            where
                Self::Scalar: core::convert::FloatToInt<I>,
            {
                // Safety: supported types are guaranteed by SimdCast, the caller is responsible for the extra invariants
                unsafe { core::intrinsics::simd::simd_cast(self) }
            }

            #[inline]
            fn to_bits(self) -> Simd<$bits_ty, N> {
                assert_eq!(core::mem::size_of::<Self>(), core::mem::size_of::<Self::Bits>());
                // Safety: transmuting between vector types is safe
                unsafe { core::mem::transmute_copy(&self) }
            }

            #[inline]
            fn from_bits(bits: Simd<$bits_ty, N>) -> Self {
                assert_eq!(core::mem::size_of::<Self>(), core::mem::size_of::<Self::Bits>());
                // Safety: transmuting between vector types is safe
                unsafe { core::mem::transmute_copy(&bits) }
            }

            #[inline]
            fn abs(self) -> Self {
                // Safety: `self` is a float vector
                unsafe { core::intrinsics::simd::simd_fabs(self) }
            }

            #[inline]
            fn recip(self) -> Self {
                Self::splat(1.0) / self
            }

            #[inline]
            fn to_degrees(self) -> Self {
                // to_degrees uses a special constant for better precision, so extract that constant
                self * Self::splat(Self::Scalar::to_degrees(1.))
            }

            #[inline]
            fn to_radians(self) -> Self {
                self * Self::splat(Self::Scalar::to_radians(1.))
            }

            #[inline]
            fn is_sign_positive(self) -> Self::Mask {
                !self.is_sign_negative()
            }

            #[inline]
            fn is_sign_negative(self) -> Self::Mask {
                let sign_bits = self.to_bits() & Simd::splat((!0 >> 1) + 1);
                sign_bits.simd_gt(Simd::splat(0))
            }

            #[inline]
            fn is_nan(self) -> Self::Mask {
                self.simd_ne(self)
            }

            #[inline]
            fn is_infinite(self) -> Self::Mask {
                self.abs().simd_eq(Self::splat(Self::Scalar::INFINITY))
            }

            #[inline]
            fn is_finite(self) -> Self::Mask {
                self.abs().simd_lt(Self::splat(Self::Scalar::INFINITY))
            }

            #[inline]
            fn is_subnormal(self) -> Self::Mask {
                // On some architectures (e.g. armv7 and some ppc) subnormals are flushed to zero,
                // so this comparison must be done with integers.
                let not_zero = self.abs().to_bits().simd_ne(Self::splat(0.0).to_bits());
                not_zero & (self.to_bits() & Self::splat(Self::Scalar::INFINITY).to_bits()).simd_eq(Simd::splat(0))
            }

            #[inline]
            #[must_use = "method returns a new mask and does not mutate the original value"]
            fn is_normal(self) -> Self::Mask {
                !(self.abs().simd_eq(Self::splat(0.0)) | self.is_nan() | self.is_subnormal() | self.is_infinite())
            }

            #[inline]
            fn signum(self) -> Self {
                self.is_nan().select(Self::splat(Self::Scalar::NAN), Self::splat(1.0).copysign(self))
            }

            #[inline]
            fn copysign(self, sign: Self) -> Self {
                let sign_bit = sign.to_bits() & Self::splat(-0.).to_bits();
                let magnitude = self.to_bits() & !Self::splat(-0.).to_bits();
                Self::from_bits(sign_bit | magnitude)
            }

            #[inline]
            fn simd_min(self, other: Self) -> Self {
                // Safety: `self` and `other` are float vectors
                unsafe { core::intrinsics::simd::simd_fmin(self, other) }
            }

            #[inline]
            fn simd_max(self, other: Self) -> Self {
                // Safety: `self` and `other` are floating point vectors
                unsafe { core::intrinsics::simd::simd_fmax(self, other) }
            }

            #[inline]
            fn simd_clamp(self, min: Self, max: Self) -> Self {
                assert!(
                    min.simd_le(max).all(),
                    "each element in `min` must be less than or equal to the corresponding element in `max`",
                );
                let mut x = self;
                x = x.simd_lt(min).select(min, x);
                x = x.simd_gt(max).select(max, x);
                x
            }

            #[inline]
            fn reduce_sum(self) -> Self::Scalar {
                // LLVM sum is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().sum()
                } else {
                    // Safety: `self` is a float vector
                    unsafe { core::intrinsics::simd::simd_reduce_add_ordered(self, -0.) }
                }
            }

            #[inline]
            fn reduce_product(self) -> Self::Scalar {
                // LLVM product is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().product()
                } else {
                    // Safety: `self` is a float vector
                    unsafe { core::intrinsics::simd::simd_reduce_mul_ordered(self, 1.) }
                }
            }

            #[inline]
            fn reduce_max(self) -> Self::Scalar {
                // Safety: `self` is a float vector
                unsafe { core::intrinsics::simd::simd_reduce_max(self) }
            }

            #[inline]
            fn reduce_min(self) -> Self::Scalar {
                // Safety: `self` is a float vector
                unsafe { core::intrinsics::simd::simd_reduce_min(self) }
            }
        }
        )*
    }
}

impl_trait! { f32 { bits: u32, mask: i32 }, f64 { bits: u64, mask: i64 } }

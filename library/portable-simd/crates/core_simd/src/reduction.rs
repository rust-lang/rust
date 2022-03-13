use crate::simd::intrinsics::{
    simd_reduce_add_ordered, simd_reduce_and, simd_reduce_max, simd_reduce_min,
    simd_reduce_mul_ordered, simd_reduce_or, simd_reduce_xor,
};
use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::ops::{BitAnd, BitOr, BitXor};

macro_rules! impl_integer_reductions {
    { $scalar:ty } => {
        impl<const LANES: usize> Simd<$scalar, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            /// Reducing wrapping add.  Returns the sum of the lanes of the vector, with wrapping addition.
            #[inline]
            pub fn reduce_sum(self) -> $scalar {
                // Safety: `self` is an integer vector
                unsafe { simd_reduce_add_ordered(self, 0) }
            }

            /// Reducing wrapping multiply.  Returns the product of the lanes of the vector, with wrapping multiplication.
            #[inline]
            pub fn reduce_product(self) -> $scalar {
                // Safety: `self` is an integer vector
                unsafe { simd_reduce_mul_ordered(self, 1) }
            }

            /// Reducing maximum.  Returns the maximum lane in the vector.
            #[inline]
            pub fn reduce_max(self) -> $scalar {
                // Safety: `self` is an integer vector
                unsafe { simd_reduce_max(self) }
            }

            /// Reducing minimum.  Returns the minimum lane in the vector.
            #[inline]
            pub fn reduce_min(self) -> $scalar {
                // Safety: `self` is an integer vector
                unsafe { simd_reduce_min(self) }
            }
        }
    }
}

impl_integer_reductions! { i8 }
impl_integer_reductions! { i16 }
impl_integer_reductions! { i32 }
impl_integer_reductions! { i64 }
impl_integer_reductions! { isize }
impl_integer_reductions! { u8 }
impl_integer_reductions! { u16 }
impl_integer_reductions! { u32 }
impl_integer_reductions! { u64 }
impl_integer_reductions! { usize }

macro_rules! impl_float_reductions {
    { $scalar:ty } => {
        impl<const LANES: usize> Simd<$scalar, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {

            /// Reducing add.  Returns the sum of the lanes of the vector.
            #[inline]
            pub fn reduce_sum(self) -> $scalar {
                // LLVM sum is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().sum()
                } else {
                    // Safety: `self` is a float vector
                    unsafe { simd_reduce_add_ordered(self, 0.) }
                }
            }

            /// Reducing multiply.  Returns the product of the lanes of the vector.
            #[inline]
            pub fn reduce_product(self) -> $scalar {
                // LLVM product is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().product()
                } else {
                    // Safety: `self` is a float vector
                    unsafe { simd_reduce_mul_ordered(self, 1.) }
                }
            }

            /// Reducing maximum.  Returns the maximum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn reduce_max(self) -> $scalar {
                // Safety: `self` is a float vector
                unsafe { simd_reduce_max(self) }
            }

            /// Reducing minimum.  Returns the minimum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn reduce_min(self) -> $scalar {
                // Safety: `self` is a float vector
                unsafe { simd_reduce_min(self) }
            }
        }
    }
}

impl_float_reductions! { f32 }
impl_float_reductions! { f64 }

impl<T, const LANES: usize> Simd<T, LANES>
where
    Self: BitAnd<Self, Output = Self>,
    T: SimdElement + BitAnd<T, Output = T>,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Reducing bitwise "and".  Returns the cumulative bitwise "and" across the lanes of
    /// the vector.
    #[inline]
    pub fn reduce_and(self) -> T {
        unsafe { simd_reduce_and(self) }
    }
}

impl<T, const LANES: usize> Simd<T, LANES>
where
    Self: BitOr<Self, Output = Self>,
    T: SimdElement + BitOr<T, Output = T>,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Reducing bitwise "or".  Returns the cumulative bitwise "or" across the lanes of
    /// the vector.
    #[inline]
    pub fn reduce_or(self) -> T {
        unsafe { simd_reduce_or(self) }
    }
}

impl<T, const LANES: usize> Simd<T, LANES>
where
    Self: BitXor<Self, Output = Self>,
    T: SimdElement + BitXor<T, Output = T>,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Reducing bitwise "xor".  Returns the cumulative bitwise "xor" across the lanes of
    /// the vector.
    #[inline]
    pub fn reduce_xor(self) -> T {
        unsafe { simd_reduce_xor(self) }
    }
}

use crate::simd::intrinsics;
use crate::simd::{LaneCount, Mask, MaskElement, Simd, SimdElement, SupportedLaneCount};

mod sealed {
    pub trait Sealed<Mask> {
        fn select(mask: Mask, true_values: Self, false_values: Self) -> Self;
    }
}
use sealed::Sealed;

/// Supporting trait for vector `select` function
pub trait Select<Mask>: Sealed<Mask> {}

impl<T, const LANES: usize> Sealed<Mask<T::Mask, LANES>> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn select(mask: Mask<T::Mask, LANES>, true_values: Self, false_values: Self) -> Self {
        unsafe { intrinsics::simd_select(mask.to_int(), true_values, false_values) }
    }
}

impl<T, const LANES: usize> Select<Mask<T::Mask, LANES>> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<T, const LANES: usize> Sealed<Self> for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn select(mask: Self, true_values: Self, false_values: Self) -> Self {
        mask & true_values | !mask & false_values
    }
}

impl<T, const LANES: usize> Select<Self> for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<T, const LANES: usize> Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Choose lanes from two vectors.
    ///
    /// For each lane in the mask, choose the corresponding lane from `true_values` if
    /// that lane mask is true, and `false_values` if that lane mask is false.
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::{Simd, Mask};
    /// # #[cfg(not(feature = "std"))] use core::simd::{Simd, Mask};
    /// let a = Simd::from_array([0, 1, 2, 3]);
    /// let b = Simd::from_array([4, 5, 6, 7]);
    /// let mask = Mask::from_array([true, false, false, true]);
    /// let c = mask.select(a, b);
    /// assert_eq!(c.to_array(), [0, 5, 6, 3]);
    /// ```
    ///
    /// `select` can also be used on masks:
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::Mask;
    /// # #[cfg(not(feature = "std"))] use core::simd::Mask;
    /// let a = Mask::<i32, 4>::from_array([true, true, false, false]);
    /// let b = Mask::<i32, 4>::from_array([false, false, true, true]);
    /// let mask = Mask::<i32, 4>::from_array([true, false, false, true]);
    /// let c = mask.select(a, b);
    /// assert_eq!(c.to_array(), [true, false, true, false]);
    /// ```
    #[inline]
    pub fn select<S: Select<Self>>(self, true_values: S, false_values: S) -> S {
        S::select(self, true_values, false_values)
    }
}

use crate::simd::intrinsics;
use crate::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};

impl<T, const LANES: usize> Simd<T, LANES>
where
    T: SimdElement + PartialEq,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Test if each lane is equal to the corresponding lane in `other`.
    #[inline]
    pub fn lanes_eq(self, other: Self) -> Mask<T::Mask, LANES> {
        unsafe { Mask::from_int_unchecked(intrinsics::simd_eq(self, other)) }
    }

    /// Test if each lane is not equal to the corresponding lane in `other`.
    #[inline]
    pub fn lanes_ne(self, other: Self) -> Mask<T::Mask, LANES> {
        unsafe { Mask::from_int_unchecked(intrinsics::simd_ne(self, other)) }
    }
}

impl<T, const LANES: usize> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Test if each lane is less than the corresponding lane in `other`.
    #[inline]
    pub fn lanes_lt(self, other: Self) -> Mask<T::Mask, LANES> {
        unsafe { Mask::from_int_unchecked(intrinsics::simd_lt(self, other)) }
    }

    /// Test if each lane is greater than the corresponding lane in `other`.
    #[inline]
    pub fn lanes_gt(self, other: Self) -> Mask<T::Mask, LANES> {
        unsafe { Mask::from_int_unchecked(intrinsics::simd_gt(self, other)) }
    }

    /// Test if each lane is less than or equal to the corresponding lane in `other`.
    #[inline]
    pub fn lanes_le(self, other: Self) -> Mask<T::Mask, LANES> {
        unsafe { Mask::from_int_unchecked(intrinsics::simd_le(self, other)) }
    }

    /// Test if each lane is greater than or equal to the corresponding lane in `other`.
    #[inline]
    pub fn lanes_ge(self, other: Self) -> Mask<T::Mask, LANES> {
        unsafe { Mask::from_int_unchecked(intrinsics::simd_ge(self, other)) }
    }
}

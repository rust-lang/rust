use super::sealed::Sealed;
use crate::simd::{intrinsics, LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};

/// Operations on SIMD vectors of constant pointers.
pub trait SimdConstPtr: Copy + Sealed {
    /// Vector type representing the pointers as bits.
    type Bits;

    /// Vector of mutable pointers to the same type.
    type MutPtr;

    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Returns `true` for each lane that is null.
    fn is_null(self) -> Self::Mask;

    /// Changes constness without changing the type.
    fn as_mut(self) -> Self::MutPtr;

    /// Cast pointers to raw bits.
    fn to_bits(self) -> Self::Bits;

    /// Cast raw bits to pointers.
    fn from_bits(bits: Self::Bits) -> Self;
}

impl<T, const LANES: usize> Sealed for Simd<*const T, LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}

impl<T, const LANES: usize> SimdConstPtr for Simd<*const T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Bits = Simd<usize, LANES>;
    type MutPtr = Simd<*mut T, LANES>;
    type Mask = Mask<isize, LANES>;

    fn is_null(self) -> Self::Mask {
        Simd::splat(core::ptr::null()).simd_eq(self)
    }

    fn as_mut(self) -> Self::MutPtr {
        // Converting between pointers is safe
        unsafe { intrinsics::simd_as(self) }
    }

    fn to_bits(self) -> Self::Bits {
        // Casting pointers to usize is safe
        unsafe { intrinsics::simd_as(self) }
    }

    fn from_bits(bits: Self::Bits) -> Self {
        // Casting usize to pointers is safe
        unsafe { intrinsics::simd_as(bits) }
    }
}

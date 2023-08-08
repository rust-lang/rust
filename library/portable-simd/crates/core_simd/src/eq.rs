use crate::simd::{
    intrinsics, LaneCount, Mask, Simd, SimdConstPtr, SimdElement, SimdMutPtr, SupportedLaneCount,
};

/// Parallel `PartialEq`.
pub trait SimdPartialEq {
    /// The mask type returned by each comparison.
    type Mask;

    /// Test if each lane is equal to the corresponding lane in `other`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn simd_eq(self, other: Self) -> Self::Mask;

    /// Test if each lane is equal to the corresponding lane in `other`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn simd_ne(self, other: Self) -> Self::Mask;
}

macro_rules! impl_number {
    { $($number:ty),* } => {
        $(
        impl<const LANES: usize> SimdPartialEq for Simd<$number, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Mask = Mask<<$number as SimdElement>::Mask, LANES>;

            #[inline]
            fn simd_eq(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Mask::from_int_unchecked(intrinsics::simd_eq(self, other)) }
            }

            #[inline]
            fn simd_ne(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Mask::from_int_unchecked(intrinsics::simd_ne(self, other)) }
            }
        }
        )*
    }
}

impl_number! { f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize }

macro_rules! impl_mask {
    { $($integer:ty),* } => {
        $(
        impl<const LANES: usize> SimdPartialEq for Mask<$integer, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Mask = Self;

            #[inline]
            fn simd_eq(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Self::from_int_unchecked(intrinsics::simd_eq(self.to_int(), other.to_int())) }
            }

            #[inline]
            fn simd_ne(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Self::from_int_unchecked(intrinsics::simd_ne(self.to_int(), other.to_int())) }
            }
        }
        )*
    }
}

impl_mask! { i8, i16, i32, i64, isize }

impl<T, const LANES: usize> SimdPartialEq for Simd<*const T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Mask = Mask<isize, LANES>;

    #[inline]
    fn simd_eq(self, other: Self) -> Self::Mask {
        self.addr().simd_eq(other.addr())
    }

    #[inline]
    fn simd_ne(self, other: Self) -> Self::Mask {
        self.addr().simd_ne(other.addr())
    }
}

impl<T, const LANES: usize> SimdPartialEq for Simd<*mut T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Mask = Mask<isize, LANES>;

    #[inline]
    fn simd_eq(self, other: Self) -> Self::Mask {
        self.addr().simd_eq(other.addr())
    }

    #[inline]
    fn simd_ne(self, other: Self) -> Self::Mask {
        self.addr().simd_ne(other.addr())
    }
}

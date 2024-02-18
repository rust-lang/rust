use crate::simd::{
    ptr::{SimdConstPtr, SimdMutPtr},
    LaneCount, Mask, Simd, SimdElement, SupportedLaneCount,
};

/// Parallel `PartialEq`.
pub trait SimdPartialEq {
    /// The mask type returned by each comparison.
    type Mask;

    /// Test if each element is equal to the corresponding element in `other`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn simd_eq(self, other: Self) -> Self::Mask;

    /// Test if each element is equal to the corresponding element in `other`.
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn simd_ne(self, other: Self) -> Self::Mask;
}

macro_rules! impl_number {
    { $($number:ty),* } => {
        $(
        impl<const N: usize> SimdPartialEq for Simd<$number, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Mask = Mask<<$number as SimdElement>::Mask, N>;

            #[inline]
            fn simd_eq(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Mask::from_int_unchecked(core::intrinsics::simd::simd_eq(self, other)) }
            }

            #[inline]
            fn simd_ne(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Mask::from_int_unchecked(core::intrinsics::simd::simd_ne(self, other)) }
            }
        }
        )*
    }
}

impl_number! { f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize }

macro_rules! impl_mask {
    { $($integer:ty),* } => {
        $(
        impl<const N: usize> SimdPartialEq for Mask<$integer, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Mask = Self;

            #[inline]
            fn simd_eq(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Self::from_int_unchecked(core::intrinsics::simd::simd_eq(self.to_int(), other.to_int())) }
            }

            #[inline]
            fn simd_ne(self, other: Self) -> Self::Mask {
                // Safety: `self` is a vector, and the result of the comparison
                // is always a valid mask.
                unsafe { Self::from_int_unchecked(core::intrinsics::simd::simd_ne(self.to_int(), other.to_int())) }
            }
        }
        )*
    }
}

impl_mask! { i8, i16, i32, i64, isize }

impl<T, const N: usize> SimdPartialEq for Simd<*const T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Mask = Mask<isize, N>;

    #[inline]
    fn simd_eq(self, other: Self) -> Self::Mask {
        self.addr().simd_eq(other.addr())
    }

    #[inline]
    fn simd_ne(self, other: Self) -> Self::Mask {
        self.addr().simd_ne(other.addr())
    }
}

impl<T, const N: usize> SimdPartialEq for Simd<*mut T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Mask = Mask<isize, N>;

    #[inline]
    fn simd_eq(self, other: Self) -> Self::Mask {
        self.addr().simd_eq(other.addr())
    }

    #[inline]
    fn simd_ne(self, other: Self) -> Self::Mask {
        self.addr().simd_ne(other.addr())
    }
}

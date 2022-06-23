use super::sealed::Sealed;
use crate::simd::{LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};

/// Operations on SIMD vectors of constant pointers.
pub trait SimdConstPtr: Copy + Sealed {
    /// Vector of usize with the same number of lanes.
    type Usize;

    /// Vector of mutable pointers to the same type.
    type MutPtr;

    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Returns `true` for each lane that is null.
    fn is_null(self) -> Self::Mask;

    /// Changes constness without changing the type.
    fn as_mut(self) -> Self::MutPtr;

    /// Gets the "address" portion of the pointer.
    ///
    /// Equivalent to calling [`pointer::addr`] on each lane.
    fn addr(self) -> Self::Usize;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_add`] on each lane.
    fn wrapping_add(self, count: Self::Usize) -> Self;
}

impl<T, const LANES: usize> Sealed for Simd<*const T, LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}

impl<T, const LANES: usize> SimdConstPtr for Simd<*const T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Usize = Simd<usize, LANES>;
    type MutPtr = Simd<*mut T, LANES>;
    type Mask = Mask<isize, LANES>;

    #[inline]
    fn is_null(self) -> Self::Mask {
        Simd::splat(core::ptr::null()).simd_eq(self)
    }

    #[inline]
    fn as_mut(self) -> Self::MutPtr {
        self.cast()
    }

    #[inline]
    fn addr(self) -> Self::Usize {
        self.cast()
    }

    #[inline]
    fn wrapping_add(self, count: Self::Usize) -> Self {
        let addr = self.addr() + (count * Simd::splat(core::mem::size_of::<T>()));
        // Safety: transmuting usize to pointers is safe, even if accessing those pointers isn't.
        unsafe { core::mem::transmute_copy(&addr) }
    }
}

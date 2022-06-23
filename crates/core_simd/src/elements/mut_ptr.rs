use super::sealed::Sealed;
use crate::simd::{LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};

/// Operations on SIMD vectors of mutable pointers.
pub trait SimdMutPtr: Copy + Sealed {
    /// Vector of usize with the same number of lanes.
    type Usize;

    /// Vector of constant pointers to the same type.
    type ConstPtr;

    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Returns `true` for each lane that is null.
    fn is_null(self) -> Self::Mask;

    /// Changes constness without changing the type.
    fn as_const(self) -> Self::ConstPtr;

    /// Gets the "address" portion of the pointer.
    ///
    /// Equivalent to calling [`pointer::addr`] on each lane.
    fn addr(self) -> Self::Usize;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_add`] on each lane.
    fn wrapping_add(self, count: Self::Usize) -> Self;
}

impl<T, const LANES: usize> Sealed for Simd<*mut T, LANES> where LaneCount<LANES>: SupportedLaneCount
{}

impl<T, const LANES: usize> SimdMutPtr for Simd<*mut T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Usize = Simd<usize, LANES>;
    type ConstPtr = Simd<*const T, LANES>;
    type Mask = Mask<isize, LANES>;

    #[inline]
    fn is_null(self) -> Self::Mask {
        Simd::splat(core::ptr::null_mut()).simd_eq(self)
    }

    #[inline]
    fn as_const(self) -> Self::ConstPtr {
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

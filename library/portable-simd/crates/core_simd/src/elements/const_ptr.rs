use super::sealed::Sealed;
use crate::simd::{intrinsics, LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};

/// Operations on SIMD vectors of constant pointers.
pub trait SimdConstPtr: Copy + Sealed {
    /// Vector of `usize` with the same number of lanes.
    type Usize;

    /// Vector of `isize` with the same number of lanes.
    type Isize;

    /// Vector of mutable pointers to the same type.
    type MutPtr;

    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Returns `true` for each lane that is null.
    fn is_null(self) -> Self::Mask;

    /// Changes constness without changing the type.
    ///
    /// Equivalent to calling [`pointer::cast_mut`] on each lane.
    fn cast_mut(self) -> Self::MutPtr;

    /// Gets the "address" portion of the pointer.
    ///
    /// This method discards pointer semantic metadata, so the result cannot be
    /// directly cast into a valid pointer.
    ///
    /// This method semantically discards *provenance* and
    /// *address-space* information. To properly restore that information, use [`Self::with_addr`].
    ///
    /// Equivalent to calling [`pointer::addr`] on each lane.
    fn addr(self) -> Self::Usize;

    /// Creates a new pointer with the given address.
    ///
    /// This performs the same operation as a cast, but copies the *address-space* and
    /// *provenance* of `self` to the new pointer.
    ///
    /// Equivalent to calling [`pointer::with_addr`] on each lane.
    fn with_addr(self, addr: Self::Usize) -> Self;

    /// Gets the "address" portion of the pointer, and "exposes" the provenance part for future use
    /// in [`Self::from_exposed_addr`].
    fn expose_addr(self) -> Self::Usize;

    /// Convert an address back to a pointer, picking up a previously "exposed" provenance.
    ///
    /// Equivalent to calling [`core::ptr::from_exposed_addr`] on each lane.
    fn from_exposed_addr(addr: Self::Usize) -> Self;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_offset`] on each lane.
    fn wrapping_offset(self, offset: Self::Isize) -> Self;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_add`] on each lane.
    fn wrapping_add(self, count: Self::Usize) -> Self;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_sub`] on each lane.
    fn wrapping_sub(self, count: Self::Usize) -> Self;
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
    type Isize = Simd<isize, LANES>;
    type MutPtr = Simd<*mut T, LANES>;
    type Mask = Mask<isize, LANES>;

    #[inline]
    fn is_null(self) -> Self::Mask {
        Simd::splat(core::ptr::null()).simd_eq(self)
    }

    #[inline]
    fn cast_mut(self) -> Self::MutPtr {
        self.cast_ptr()
    }

    #[inline]
    fn addr(self) -> Self::Usize {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        // SAFETY: Pointer-to-integer transmutes are valid (if you are okay with losing the
        // provenance).
        unsafe { core::mem::transmute_copy(&self) }
    }

    #[inline]
    fn with_addr(self, addr: Self::Usize) -> Self {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        //
        // In the mean-time, this operation is defined to be "as if" it was
        // a wrapping_offset, so we can emulate it as such. This should properly
        // restore pointer provenance even under today's compiler.
        self.cast_ptr::<*const u8>()
            .wrapping_offset(addr.cast::<isize>() - self.addr().cast::<isize>())
            .cast_ptr()
    }

    #[inline]
    fn expose_addr(self) -> Self::Usize {
        // Safety: `self` is a pointer vector
        unsafe { intrinsics::simd_expose_addr(self) }
    }

    #[inline]
    fn from_exposed_addr(addr: Self::Usize) -> Self {
        // Safety: `self` is a pointer vector
        unsafe { intrinsics::simd_from_exposed_addr(addr) }
    }

    #[inline]
    fn wrapping_offset(self, count: Self::Isize) -> Self {
        // Safety: simd_arith_offset takes a vector of pointers and a vector of offsets
        unsafe { intrinsics::simd_arith_offset(self, count) }
    }

    #[inline]
    fn wrapping_add(self, count: Self::Usize) -> Self {
        self.wrapping_offset(count.cast())
    }

    #[inline]
    fn wrapping_sub(self, count: Self::Usize) -> Self {
        self.wrapping_offset(-count.cast::<isize>())
    }
}

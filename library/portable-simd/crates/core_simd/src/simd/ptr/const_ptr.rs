use super::sealed::Sealed;
use crate::simd::{cmp::SimdPartialEq, num::SimdUint, LaneCount, Mask, Simd, SupportedLaneCount};

/// Operations on SIMD vectors of constant pointers.
pub trait SimdConstPtr: Copy + Sealed {
    /// Vector of `usize` with the same number of elements.
    type Usize;

    /// Vector of `isize` with the same number of elements.
    type Isize;

    /// Vector of const pointers with the same number of elements.
    type CastPtr<T>;

    /// Vector of mutable pointers to the same type.
    type MutPtr;

    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Returns `true` for each element that is null.
    fn is_null(self) -> Self::Mask;

    /// Casts to a pointer of another type.
    ///
    /// Equivalent to calling [`pointer::cast`] on each element.
    fn cast<T>(self) -> Self::CastPtr<T>;

    /// Changes constness without changing the type.
    ///
    /// Equivalent to calling [`pointer::cast_mut`] on each element.
    fn cast_mut(self) -> Self::MutPtr;

    /// Gets the "address" portion of the pointer.
    ///
    /// This method discards pointer semantic metadata, so the result cannot be
    /// directly cast into a valid pointer.
    ///
    /// This method semantically discards *provenance* and
    /// *address-space* information. To properly restore that information, use [`Self::with_addr`].
    ///
    /// Equivalent to calling [`pointer::addr`] on each element.
    fn addr(self) -> Self::Usize;

    /// Creates a new pointer with the given address.
    ///
    /// This performs the same operation as a cast, but copies the *address-space* and
    /// *provenance* of `self` to the new pointer.
    ///
    /// Equivalent to calling [`pointer::with_addr`] on each element.
    fn with_addr(self, addr: Self::Usize) -> Self;

    /// Exposes the "provenance" part of the pointer for future use in
    /// [`Self::with_exposed_provenance`] and returns the "address" portion.
    fn expose_provenance(self) -> Self::Usize;

    /// Converts an address back to a pointer, picking up a previously "exposed" provenance.
    ///
    /// Equivalent to calling [`core::ptr::with_exposed_provenance`] on each element.
    fn with_exposed_provenance(addr: Self::Usize) -> Self;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_offset`] on each element.
    fn wrapping_offset(self, offset: Self::Isize) -> Self;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_add`] on each element.
    fn wrapping_add(self, count: Self::Usize) -> Self;

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// Equivalent to calling [`pointer::wrapping_sub`] on each element.
    fn wrapping_sub(self, count: Self::Usize) -> Self;
}

impl<T, const N: usize> Sealed for Simd<*const T, N> where LaneCount<N>: SupportedLaneCount {}

impl<T, const N: usize> SimdConstPtr for Simd<*const T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Usize = Simd<usize, N>;
    type Isize = Simd<isize, N>;
    type CastPtr<U> = Simd<*const U, N>;
    type MutPtr = Simd<*mut T, N>;
    type Mask = Mask<isize, N>;

    #[inline]
    fn is_null(self) -> Self::Mask {
        Simd::splat(core::ptr::null()).simd_eq(self)
    }

    #[inline]
    fn cast<U>(self) -> Self::CastPtr<U> {
        // SimdElement currently requires zero-sized metadata, so this should never fail.
        // If this ever changes, `simd_cast_ptr` should produce a post-mono error.
        use core::ptr::Pointee;
        assert_eq!(size_of::<<T as Pointee>::Metadata>(), 0);
        assert_eq!(size_of::<<U as Pointee>::Metadata>(), 0);

        // Safety: pointers can be cast
        unsafe { core::intrinsics::simd::simd_cast_ptr(self) }
    }

    #[inline]
    fn cast_mut(self) -> Self::MutPtr {
        // Safety: pointers can be cast
        unsafe { core::intrinsics::simd::simd_cast_ptr(self) }
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
        self.cast::<u8>()
            .wrapping_offset(addr.cast::<isize>() - self.addr().cast::<isize>())
            .cast()
    }

    #[inline]
    fn expose_provenance(self) -> Self::Usize {
        // Safety: `self` is a pointer vector
        unsafe { core::intrinsics::simd::simd_expose_provenance(self) }
    }

    #[inline]
    fn with_exposed_provenance(addr: Self::Usize) -> Self {
        // Safety: `self` is a pointer vector
        unsafe { core::intrinsics::simd::simd_with_exposed_provenance(addr) }
    }

    #[inline]
    fn wrapping_offset(self, count: Self::Isize) -> Self {
        // Safety: simd_arith_offset takes a vector of pointers and a vector of offsets
        unsafe { core::intrinsics::simd::simd_arith_offset(self, count) }
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

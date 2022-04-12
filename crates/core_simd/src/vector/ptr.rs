//! Private implementation details of public gather/scatter APIs.
#[cfg(not(bootstrap))]
use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SupportedLaneCount};
#[cfg(bootstrap)]
use core::mem;

/// A vector of *const T.
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub(crate) struct SimdConstPtr<T, const LANES: usize>([*const T; LANES]);

impl<T, const LANES: usize> SimdConstPtr<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: Sized,
{
    #[inline]
    #[must_use]
    pub fn splat(ptr: *const T) -> Self {
        Self([ptr; LANES])
    }

    #[inline]
    #[must_use]
    pub fn wrapping_add(self, addend: Simd<usize, LANES>) -> Self {
        #[cfg(bootstrap)]
        // Safety: converting pointers to usize and vice-versa is safe
        // (even if using that pointer is not)
        unsafe {
            let x: Simd<usize, LANES> = mem::transmute_copy(&self);
            mem::transmute_copy(&{ x + (addend * Simd::splat(mem::size_of::<T>())) })
        }
        #[cfg(not(bootstrap))]
        // Safety: this intrinsic doesn't have a precondition
        unsafe { intrinsics::simd_arith_offset(self, addend) }
    }
}

/// A vector of *mut T. Be very careful around potential aliasing.
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub(crate) struct SimdMutPtr<T, const LANES: usize>([*mut T; LANES]);

impl<T, const LANES: usize> SimdMutPtr<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: Sized,
{
    #[inline]
    #[must_use]
    pub fn splat(ptr: *mut T) -> Self {
        Self([ptr; LANES])
    }

    #[inline]
    #[must_use]
    pub fn wrapping_add(self, addend: Simd<usize, LANES>) -> Self {
        #[cfg(bootstrap)]
        // Safety: converting pointers to usize and vice-versa is safe
        // (even if using that pointer is not)
        unsafe {
            let x: Simd<usize, LANES> = mem::transmute_copy(&self);
            mem::transmute_copy(&{ x + (addend * Simd::splat(mem::size_of::<T>())) })
        }
        #[cfg(not(bootstrap))]
        // Safety: this intrinsic doesn't have a precondition
        unsafe { intrinsics::simd_arith_offset(self, addend) }
    }
}

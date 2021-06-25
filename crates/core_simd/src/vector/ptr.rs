//! Private implementation details of public gather/scatter APIs.
use crate::SimdUsize;
use core::mem;

/// A vector of *const T.
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub(crate) struct SimdConstPtr<T, const LANES: usize>([*const T; LANES]);

impl<T, const LANES: usize> SimdConstPtr<T, LANES>
where
    SimdUsize<LANES>: crate::Vector,
    T: Sized,
{
    #[inline]
    #[must_use]
    pub fn splat(ptr: *const T) -> Self {
        Self([ptr; LANES])
    }

    #[inline]
    #[must_use]
    pub fn wrapping_add(self, addend: SimdUsize<LANES>) -> Self {
        unsafe {
            let x: SimdUsize<LANES> = mem::transmute_copy(&self);
            mem::transmute_copy(&{ x + (addend * mem::size_of::<T>()) })
        }
    }
}

/// A vector of *mut T. Be very careful around potential aliasing.
#[derive(Debug, Copy, Clone)]
#[repr(simd)]
pub(crate) struct SimdMutPtr<T, const LANES: usize>([*mut T; LANES]);

impl<T, const LANES: usize> SimdMutPtr<T, LANES>
where
    SimdUsize<LANES>: crate::Vector,
    T: Sized,
{
    #[inline]
    #[must_use]
    pub fn splat(ptr: *mut T) -> Self {
        Self([ptr; LANES])
    }

    #[inline]
    #[must_use]
    pub fn wrapping_add(self, addend: SimdUsize<LANES>) -> Self {
        unsafe {
            let x: SimdUsize<LANES> = mem::transmute_copy(&self);
            mem::transmute_copy(&{ x + (addend * mem::size_of::<T>()) })
        }
    }
}

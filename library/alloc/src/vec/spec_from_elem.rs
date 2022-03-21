use core::ptr;

use crate::alloc::Allocator;
use crate::raw_vec::RawVec;

use super::{ExtendElement, IsZero, Vec, VecError};

// Specialization trait used for Vec::from_elem
pub(super) trait SpecFromElem: Sized {
    fn from_elem<A: Allocator, TError: VecError>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Result<Vec<Self, A>, TError>;
}

impl<T: Clone> SpecFromElem for T {
    default fn from_elem<A: Allocator, TError: VecError>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Result<Vec<Self, A>, TError> {
        let mut v = Vec::with_capacity_in_impl(n, alloc)?;
        v.extend_with_impl(n, ExtendElement(elem))?;
        Ok(v)
    }
}

impl<T: Clone + IsZero> SpecFromElem for T {
    #[inline]
    default fn from_elem<A: Allocator, TError: VecError>(
        elem: T,
        n: usize,
        alloc: A,
    ) -> Result<Vec<T, A>, TError> {
        if elem.is_zero() {
            return Ok(Vec { buf: RawVec::with_capacity_zeroed_in_impl(n, alloc)?, len: n });
        }
        let mut v = Vec::with_capacity_in_impl(n, alloc)?;
        v.extend_with_impl(n, ExtendElement(elem))?;
        Ok(v)
    }
}

impl SpecFromElem for i8 {
    #[inline]
    fn from_elem<A: Allocator, TError: VecError>(
        elem: i8,
        n: usize,
        alloc: A,
    ) -> Result<Vec<i8, A>, TError> {
        if elem == 0 {
            return Ok(Vec { buf: RawVec::with_capacity_zeroed_in_impl(n, alloc)?, len: n });
        }
        unsafe {
            let mut v = Vec::with_capacity_in_impl(n, alloc)?;
            ptr::write_bytes(v.as_mut_ptr(), elem as u8, n);
            v.set_len(n);
            Ok(v)
        }
    }
}

impl SpecFromElem for u8 {
    #[inline]
    fn from_elem<A: Allocator, TError: VecError>(
        elem: u8,
        n: usize,
        alloc: A,
    ) -> Result<Vec<u8, A>, TError> {
        if elem == 0 {
            return Ok(Vec { buf: RawVec::with_capacity_zeroed_in_impl(n, alloc)?, len: n });
        }
        unsafe {
            let mut v = Vec::with_capacity_in_impl(n, alloc)?;
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
            Ok(v)
        }
    }
}

use core::ptr;

use crate::alloc::Allocator;
use crate::collections::TryReserveError;

use super::{IsZero, Vec};

// Specialization trait used for Vec::from_elem
pub(super) trait SpecFromElem: Sized {
    fn from_elem<A: Allocator>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Result<Vec<Self, A>, TryReserveError>;
}

impl<T: Clone> SpecFromElem for T {
    default fn from_elem<A: Allocator>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Result<Vec<Self, A>, TryReserveError> {
        let mut v = Vec::try_with_capacity_in(n, alloc)?;
        v.extend_with(n, elem)?;
        Ok(v)
    }
}

impl<T: Clone + IsZero> SpecFromElem for T {
    #[inline]
    default fn from_elem<A: Allocator>(
        elem: T,
        n: usize,
        alloc: A,
    ) -> Result<Vec<T, A>, TryReserveError> {
        if elem.is_zero() {
            let mut v = Vec::try_with_capacity_zeroed_in(n, alloc)?;
            unsafe { v.set_len(n) };
            return Ok(v);
        }
        let mut v = Vec::try_with_capacity_in(n, alloc)?;
        v.extend_with(n, elem)?;
        Ok(v)
    }
}

impl SpecFromElem for i8 {
    #[inline]
    fn from_elem<A: Allocator>(
        elem: i8,
        n: usize,
        alloc: A,
    ) -> Result<Vec<i8, A>, TryReserveError> {
        if elem == 0 {
            let mut v = Vec::try_with_capacity_zeroed_in(n, alloc)?;
            unsafe { v.set_len(n) };
            return Ok(v);
        }
        unsafe {
            let mut v = Vec::try_with_capacity_in(n, alloc)?;
            ptr::write_bytes(v.as_mut_ptr(), elem as u8, n);
            v.set_len(n);
            Ok(v)
        }
    }
}

impl SpecFromElem for u8 {
    #[inline]
    fn from_elem<A: Allocator>(
        elem: u8,
        n: usize,
        alloc: A,
    ) -> Result<Vec<u8, A>, TryReserveError> {
        if elem == 0 {
            let mut v = Vec::try_with_capacity_zeroed_in(n, alloc)?;
            unsafe { v.set_len(n) };
            return Ok(v);
        }
        unsafe {
            let mut v = Vec::try_with_capacity_in(n, alloc)?;
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
            Ok(v)
        }
    }
}

use core::ptr;

use crate::alloc::Allocator;
use crate::co_alloc::CoAllocPref;
use crate::raw_vec::RawVec;

use super::{IsZero, Vec};

// Specialization trait used for Vec::from_elem
pub(super) trait SpecFromElem: Sized {
    #[allow(unused_braces)]
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Vec<Self, A, CO_ALLOC_PREF>
    where
        [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:;
}

#[allow(unused_braces)]
impl<T: Clone> SpecFromElem for T {
    default fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Vec<Self, A, CO_ALLOC_PREF>
    where
        [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
    {
        let mut v = Vec::with_capacity_in_co(n, alloc);
        v.extend_with(n, elem);
        v
    }
}

#[allow(unused_braces)]
impl<T: Clone + IsZero> SpecFromElem for T {
    #[inline]
    default fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: T,
        n: usize,
        alloc: A,
    ) -> Vec<T, A, CO_ALLOC_PREF>
    where
        [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
    {
        if elem.is_zero() {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        let mut v = Vec::with_capacity_in_co(n, alloc);
        v.extend_with(n, elem);
        v
    }
}

impl SpecFromElem for i8 {
    #[inline]
    #[allow(unused_braces)]
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: i8,
        n: usize,
        alloc: A,
    ) -> Vec<i8, A, CO_ALLOC_PREF>
    where
        [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
    {
        if elem == 0 {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        let mut v = Vec::with_capacity_in_co(n, alloc);
        unsafe {
            ptr::write_bytes(v.as_mut_ptr(), elem as u8, n);
            v.set_len(n);
        }
        v
    }
}

impl SpecFromElem for u8 {
    #[inline]
    #[allow(unused_braces)]
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: u8,
        n: usize,
        alloc: A,
    ) -> Vec<u8, A, CO_ALLOC_PREF>
    where
        [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
    {
        if elem == 0 {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        let mut v = Vec::with_capacity_in_co(n, alloc);
        unsafe {
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
        }
        v
    }
}

// A better way would be to implement this for all ZSTs which are `Copy` and have trivial `Clone`
// but the latter cannot be detected currently
impl SpecFromElem for () {
    #[inline]
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        _elem: (),
        n: usize,
        alloc: A,
    ) -> Vec<(), A, CO_ALLOC_PREF> {
        let mut v = Vec::with_capacity_in_co(n, alloc);
        // SAFETY: the capacity has just been set to `n`
        // and `()` is a ZST with trivial `Clone` implementation
        unsafe {
            v.set_len(n);
        }
        v
    }
}

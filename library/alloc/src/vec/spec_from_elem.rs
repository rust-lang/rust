use core::ptr;

use crate::alloc::Allocator;
use crate::co_alloc::CoAllocPref;
use crate::raw_vec::RawVec;
use core::alloc;

use super::{ExtendElement, IsZero, Vec};

// Specialization trait used for Vec::from_elem
pub(super) trait SpecFromElem: Sized {
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Vec<Self, A, CO_ALLOC_PREF>
    where
        [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(CO_ALLOC_PREF)]:;
}

impl<T: Clone> SpecFromElem for T {
    default fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: Self,
        n: usize,
        alloc: A,
    ) -> Vec<Self, A, CO_ALLOC_PREF>
    where
        [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(CO_ALLOC_PREF)]:,
    {
        let mut v = Vec::with_capacity_in(n, alloc);
        v.extend_with(n, ExtendElement(elem));
        v
    }
}

impl<T: Clone + IsZero> SpecFromElem for T {
    #[inline]
    default fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: T,
        n: usize,
        alloc: A,
    ) -> Vec<T, A, CO_ALLOC_PREF>
    where
        [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(CO_ALLOC_PREF)]:,
    {
        if elem.is_zero() {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        let mut v = Vec::with_capacity_in(n, alloc);
        v.extend_with(n, ExtendElement(elem));
        v
    }
}

impl SpecFromElem for i8 {
    #[inline]
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: i8,
        n: usize,
        alloc: A,
    ) -> Vec<i8, A, CO_ALLOC_PREF>
    where
        [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(CO_ALLOC_PREF)]:,
    {
        if elem == 0 {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        unsafe {
            let mut v = Vec::with_capacity_in(n, alloc);
            ptr::write_bytes(v.as_mut_ptr(), elem as u8, n);
            v.set_len(n);
            v
        }
    }
}

impl SpecFromElem for u8 {
    #[inline]
    fn from_elem<A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
        elem: u8,
        n: usize,
        alloc: A,
    ) -> Vec<u8, A, CO_ALLOC_PREF>
    where
        [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(CO_ALLOC_PREF)]:,
    {
        if elem == 0 {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        unsafe {
            let mut v = Vec::with_capacity_in(n, alloc);
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
            v
        }
    }
}

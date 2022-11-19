use core::iter;

use crate::alloc::Allocator;
use crate::raw_vec::RawVec;

use super::{IsZero, Vec};

// Specialization trait used for Vec::from_elem
pub(super) trait SpecFromElem: Sized {
    fn from_elem<A: Allocator>(elem: Self, n: usize, alloc: A) -> Vec<Self, A>;
}

fn basic_from_elem<T: Clone, A: Allocator>(elem: T, n: usize, alloc: A) -> Vec<T, A> {
    let mut v = Vec::with_capacity_in(n, alloc);
    v.extend(iter::repeat_n(elem, n));
    v
}

impl<T: Clone> SpecFromElem for T {
    default fn from_elem<A: Allocator>(elem: Self, n: usize, alloc: A) -> Vec<Self, A> {
        basic_from_elem(elem, n, alloc)
    }
}

impl<T: Clone + IsZero> SpecFromElem for T {
    #[inline]
    fn from_elem<A: Allocator>(elem: T, n: usize, alloc: A) -> Vec<T, A> {
        if elem.is_zero() {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        basic_from_elem(elem, n, alloc)
    }
}

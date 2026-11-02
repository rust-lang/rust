use core::ptr;

use super::{IsZero, Vec};
use crate::alloc::Allocator;
use crate::raw_vec::RawVec;

// Specialization trait used for Vec::from_elem
pub(super) trait SpecFromElem: Sized {
    fn from_elem<A: Allocator>(elem: Self, n: usize, alloc: A) -> Vec<Self, A>;
}

impl<T: Clone> SpecFromElem for T {
    fn from_elem<A: Allocator>(elem: Self, n: usize, alloc: A) -> Vec<Self, A> {
        // specialization for `T: IsZero`
        if let Some(elem) = core::any::try_as_dyn::<T, dyn IsZero>(&elem)
            && elem.is_zero()
        {
            return Vec { buf: RawVec::with_capacity_zeroed_in(n, alloc), len: n };
        }
        // specialization for `u8` and `i8`
        if let Some(elem) = core::any::try_as_dyn::<T, dyn TransmutableToU8>(&elem) {
            let mut v = Vec::with_capacity_in(n, alloc);
            // ignore-tidy-undocumented-unsafe
            unsafe {
                ptr::write_bytes(v.as_mut_ptr(), elem.to_u8(), n);
                v.set_len(n);
            }
            return v;
        }
        // specialization for ZST
        if let Some(_) = core::any::try_as_dyn::<T, dyn ZstCopyAndTrivialClone>(&elem) {
            let mut v = Vec::with_capacity_in(n, alloc);
            // SAFETY: the capacity has just been set to `n`
            // and `()` is a ZST with trivial `Clone` implementation
            unsafe {
                v.set_len(n);
            }
            return v;
        }
        let mut v = Vec::with_capacity_in(n, alloc);
        v.extend_with(n, elem);
        v
    }
}

/// # Safety
/// Must be transmutable to `u8`
unsafe trait TransmutableToU8 {
    fn to_u8(&self) -> u8;
}

unsafe impl TransmutableToU8 for i8 {
    fn to_u8(&self) -> u8 {
        self.cast_unsigned()
    }
}

unsafe impl TransmutableToU8 for u8 {
    fn to_u8(&self) -> u8 {
        *self
    }
}

/// # Safety
/// Must be ZST, `Copy`, and have trivial `Clone`
unsafe trait ZstCopyAndTrivialClone {}

// A better way would be to implement this for all ZSTs which are `Copy` and have trivial `Clone`
// but the latter cannot be detected currently
unsafe impl ZstCopyAndTrivialClone for () {}

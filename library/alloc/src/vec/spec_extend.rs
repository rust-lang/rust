use crate::alloc::Allocator;
use crate::co_alloc::CoAllocPref;
use core::iter::TrustedLen;
use core::slice::{self};

use super::{IntoIter, Vec};

// Specialization trait used for Vec::extend
pub(super) trait SpecExtend<T, I> {
    fn spec_extend(&mut self, iter: I);
}

#[allow(unused_braces)]
impl<T, I, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> SpecExtend<T, I>
    for Vec<T, A, CO_ALLOC_PREF>
where
    I: Iterator<Item = T>,
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    default fn spec_extend(&mut self, iter: I) {
        self.extend_desugared(iter)
    }
}

#[allow(unused_braces)]
impl<T, I, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> SpecExtend<T, I>
    for Vec<T, A, CO_ALLOC_PREF>
where
    I: TrustedLen<Item = T>,
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    default fn spec_extend(&mut self, iterator: I) {
        self.extend_trusted(iterator)
    }
}

#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> SpecExtend<T, IntoIter<T>>
    for Vec<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn spec_extend(&mut self, mut iterator: IntoIter<T>) {
        unsafe {
            self.append_elements(iterator.as_slice() as _);
        }
        iterator.forget_remaining_elements();
    }
}

#[allow(unused_braces)]
impl<'a, T: 'a, I, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> SpecExtend<&'a T, I>
    for Vec<T, A, CO_ALLOC_PREF>
where
    I: Iterator<Item = &'a T>,
    T: Clone,
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.cloned())
    }
}

#[allow(unused_braces)]
impl<'a, T: 'a, A: Allocator, const CO_ALLOC_PREF: CoAllocPref>
    SpecExtend<&'a T, slice::Iter<'a, T>> for Vec<T, A, CO_ALLOC_PREF>
where
    T: Copy,
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn spec_extend(&mut self, iterator: slice::Iter<'a, T>) {
        let slice = iterator.as_slice();
        unsafe { self.append_elements(slice) };
    }
}

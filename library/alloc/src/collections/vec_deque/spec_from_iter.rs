use super::{IntoIter, VecDeque};
use crate::alloc::Global;
use crate::co_alloc::CoAllocPref;
use core::alloc;

/// Specialization trait used for `VecDeque::from_iter`
pub(super) trait SpecFromIter<T, I> {
    fn spec_from_iter(iter: I) -> Self;
}

impl<T, I, const CO_ALLOC_PREF: CoAllocPref> SpecFromIter<T, I> for VecDeque<T, Global, CO_ALLOC_PREF>
where
    I: Iterator<Item = T>,
    [(); alloc::co_alloc_metadata_num_slots_with_preference::<Global>(CO_ALLOC_PREF)]:,
{
    default fn spec_from_iter(iterator: I) -> Self {
        // Since converting is O(1) now, just re-use the `Vec` logic for
        // anything where we can't do something extra-special for `VecDeque`,
        // especially as that could save us some monomorphiziation work
        // if one uses the same iterators (like slice ones) with both.
        crate::vec::Vec::<T, Global, CO_ALLOC_PREF>::from_iter(iterator).into()
    }
}

impl<T, const CO_ALLOC_PREF: CoAllocPref> SpecFromIter<T, crate::vec::IntoIter<T, Global, CO_ALLOC_PREF>>
    for VecDeque<T, Global, CO_ALLOC_PREF>
where
    [(); alloc::co_alloc_metadata_num_slots_with_preference::<Global>(CO_ALLOC_PREF)]:,
{
    #[inline]
    fn spec_from_iter(iterator: crate::vec::IntoIter<T, Global, CO_ALLOC_PREF>) -> Self {
        iterator.into_vecdeque()
    }
}

impl<T, const CO_ALLOC_PREF: CoAllocPref> SpecFromIter<T, IntoIter<T, Global, CO_ALLOC_PREF>>
    for VecDeque<T, Global, CO_ALLOC_PREF>
where
    [(); alloc::co_alloc_metadata_num_slots_with_preference::<Global>(CO_ALLOC_PREF)]:,
{
    #[inline]
    fn spec_from_iter(iterator: IntoIter<T, Global, CO_ALLOC_PREF>) -> Self {
        iterator.into_vecdeque()
    }
}

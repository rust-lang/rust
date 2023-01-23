use super::{IntoIter, VecDeque};
use crate::alloc::Allocator;
use core::alloc;

/// Specialization trait used for `VecDeque::from_iter`
pub(super) trait SpecFromIter<T, I> {
    fn spec_from_iter(iter: I) -> Self;
}

impl<T, I, A: Allocator, const COOP_PREFERRED: bool> SpecFromIter<T, I>
    for VecDeque<T, A, COOP_PREFERRED>
where
    I: Iterator<Item = T>,
    [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]:,
{
    default fn spec_from_iter(iterator: I) -> Self {
        // Since converting is O(1) now, just re-use the `Vec` logic for
        // anything where we can't do something extra-special for `VecDeque`,
        // especially as that could save us some monomorphiziation work
        // if one uses the same iterators (like slice ones) with both.
        crate::vec::Vec::from_iter(iterator).into()
    }
}

impl<T, A: Allocator, const COOP_PREFERRED: bool> SpecFromIter<T, crate::vec::IntoIter<T>>
    for VecDeque<T, A, COOP_PREFERRED>
where
    [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]:,
{
    #[inline]
    fn spec_from_iter(iterator: crate::vec::IntoIter<T>) -> Self {
        iterator.into_vecdeque()
    }
}

impl<T, A: Allocator, const COOP_PREFERRED: bool> SpecFromIter<T, IntoIter<T>>
    for VecDeque<T, A, COOP_PREFERRED>
where
    [(); alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]:,
{
    #[inline]
    fn spec_from_iter(iterator: IntoIter<T>) -> Self {
        iterator.into_vecdeque()
    }
}

use super::{IntoIter, VecDeque};
use crate::alloc::Global;
use crate::co_alloc::CoAllocPref;

/// Specialization trait used for `VecDeque::from_iter`
pub(super) trait SpecFromIter<T, I> {
    fn spec_from_iter(iter: I) -> Self;
}

/// Specialization trait used for `VecDeque::from_iter_co`
pub(super) trait SpecFromIterCo<T, I> {
    fn spec_from_iter_co(iter: I) -> Self;
}

impl<T, I> SpecFromIter<T, I> for VecDeque<T>
where
    I: Iterator<Item = T>,
{
    default fn spec_from_iter(iterator: I) -> Self {
        // Since converting is O(1) now, just re-use the `Vec` logic for
        // anything where we can't do something extra-special for `VecDeque`,
        // especially as that could save us some monomorphization work
        // if one uses the same iterators (like slice ones) with both.
        crate::vec::Vec::from_iter(iterator).into()
    }
}

impl<T> SpecFromIter<T, crate::vec::IntoIter<T>> for VecDeque<T> {
    #[inline]
    fn spec_from_iter(iterator: crate::vec::IntoIter<T>) -> Self {
        iterator.into_vecdeque()
    }
}

impl<T> SpecFromIter<T, IntoIter<T>> for VecDeque<T> {
    #[inline]
    fn spec_from_iter(iterator: IntoIter<T>) -> Self {
        iterator.into_vecdeque()
    }
}
// ---- CoAllocation:

#[allow(unused_braces)]
impl<T, I, const CO_ALLOC_PREF: CoAllocPref> SpecFromIterCo<T, I>
    for VecDeque<T, Global, CO_ALLOC_PREF>
where
    I: Iterator<Item = T>,
    [(); { crate::meta_num_slots_global!(CO_ALLOC_PREF) }]:,
{
    default fn spec_from_iter_co(iterator: I) -> Self {
        // @FIXME Move the assert to library/alloc/src/macros.rs -> co_alloc_pref!(...) and replace
        // calls to CO_ALLOC_PREF_META_YES and CO_ALLOC_PREF_META_NO with constants - once
        // https://github.com/rust-lang/rust/issues/106994 (the ICE) is fixed. Upvote it, please.
        core::debug_assert!(
            CO_ALLOC_PREF == crate::CO_ALLOC_PREF_META_YES!()
                || CO_ALLOC_PREF == crate::CO_ALLOC_PREF_META_NO!(),
            "CO_ALLOC_PREF must equal to CO_ALLOC_PREF_META_YES!() or CO_ALLOC_PREF_META_NO!(), but it is: {CO_ALLOC_PREF}."
        );
        // Since converting is O(1) now, just re-use the `Vec` logic for
        // anything where we can't do something extra-special for `VecDeque`,
        // especially as that could save us some monomorphiziation work
        // if one uses the same iterators (like slice ones) with both.
        crate::vec::Vec::<T, Global, CO_ALLOC_PREF>::from_iter_co(iterator).into()
    }
}

// Until we can use feature `specialization`:
// @FIXME new macro + replace 0 and 1 with META ZERO/ONE
impl<T> SpecFromIterCo<T, crate::vec::IntoIter<T, Global, 0>> for VecDeque<T, Global, 0> {
    #[inline]
    fn spec_from_iter_co(iterator: crate::vec::IntoIter<T, Global, 0>) -> Self {
        iterator.into_vecdeque()
    }
}
impl<T> SpecFromIterCo<T, crate::vec::IntoIter<T, Global, 1>> for VecDeque<T, Global, 1> {
    #[inline]
    fn spec_from_iter_co(iterator: crate::vec::IntoIter<T, Global, 1>) -> Self {
        iterator.into_vecdeque()
    }
}

impl<T> SpecFromIterCo<T, IntoIter<T, Global, 0>> for VecDeque<T, Global, 0> {
    #[inline]
    fn spec_from_iter_co(iterator: IntoIter<T, Global, 0>) -> Self {
        iterator.into_vecdeque()
    }
}
impl<T> SpecFromIterCo<T, IntoIter<T, Global, 1>> for VecDeque<T, Global, 1> {
    #[inline]
    fn spec_from_iter_co(iterator: IntoIter<T, Global, 1>) -> Self {
        iterator.into_vecdeque()
    }
}

// WITH unstable feature `specialization`:
/*
#[allow(unused_braces)]
impl<T, const CO_ALLOC_PREF: CoAllocPref>
    SpecFromIterCo<T, crate::vec::IntoIter<T, Global, CO_ALLOC_PREF>>
    for VecDeque<T, Global, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots_global!(CO_ALLOC_PREF) }]:,
{
    #[inline]
    fn spec_from_iter_co(iterator: crate::vec::IntoIter<T, Global, CO_ALLOC_PREF>) -> Self {
        iterator.into_vecdeque()
    }
}

#[allow(unused_braces)]
impl<T, const CO_ALLOC_PREF: CoAllocPref> SpecFromIterCo<T, IntoIter<T, Global, CO_ALLOC_PREF>>
    for VecDeque<T, Global, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots_global!(CO_ALLOC_PREF) }]:,
{
    #[inline]
    fn spec_from_iter_co(iterator: IntoIter<T, Global, CO_ALLOC_PREF>) -> Self {
        iterator.into_vecdeque()
    }
}
*/

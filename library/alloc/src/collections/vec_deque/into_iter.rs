use crate::co_alloc::CoAllocPref;
use core::fmt;
use core::iter::{FusedIterator, TrustedLen};

use crate::alloc::{Allocator, Global};

use super::VecDeque;

/// An owning iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`into_iter`] method on [`VecDeque`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: VecDeque::into_iter
/// [`IntoIterator`]: core::iter::IntoIterator
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
pub struct IntoIter<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
    const CO_ALLOC_PREF: CoAllocPref = { CO_ALLOC_PREF_DEFAULT!() },
> where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    inner: VecDeque<T, A, CO_ALLOC_PREF>,
}

#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    pub(super) fn new(inner: VecDeque<T, A, CO_ALLOC_PREF>) -> Self {
        IntoIter { inner }
    }

    pub(super) fn into_vecdeque(self) -> VecDeque<T, A, CO_ALLOC_PREF> {
        self.inner
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
#[allow(unused_braces)]
impl<T: fmt::Debug, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> fmt::Debug
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Iterator for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> DoubleEndedIterator
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> ExactSizeIterator
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> FusedIterator
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
#[allow(unused_braces)]
unsafe impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> TrustedLen
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}

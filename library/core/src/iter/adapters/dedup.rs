#![unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]

use crate::iter::{InPlaceIterable, SourceIter};

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator according to the [`PartialEq`] trait implementation.
///
/// This `struct` is created by [`Iterator::dedup`].
/// See its documentation for more.
///
/// [`Iterator::dedup`]: Iterator::dedup
#[derive(Debug, Clone)]
pub struct Dedup<I>
where
    I: Iterator,
{
    inner: I,
    last: Option<Option<I::Item>>,
}

impl<I> Dedup<I>
where
    I: Iterator,
{
    #[inline]
    pub(crate) fn new(inner: I) -> Self {
        Self { inner, last: None }
    }
}

impl<I> Iterator for Dedup<I>
where
    I: Iterator,
    I::Item: PartialEq,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { inner, last } = self;
        let last = last.get_or_insert_with(|| inner.next());
        let last_item = last.as_ref()?;
        let next = inner.find(|next_item| next_item != last_item);
        crate::mem::replace(last, next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = matches!(self.last, Some(Some(_))).into();
        let max = self.inner.size_hint().1.map(|max| max + min);
        (min, max)
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> SourceIter for Dedup<I>
where
    I: SourceIter + Iterator,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.inner) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> InPlaceIterable for Dedup<I>
where
    I: Iterator,
    I: InPlaceIterable,
    I::Item: PartialEq,
{
}

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator satisfying a given equality relation.
///
/// This `struct` is created by [`Iterator::dedup_by`].
/// See its documentation for more.
///
/// [`Iterator::dedup_by`]: Iterator::dedup_by
#[derive(Debug, Clone)]
pub struct DedupBy<I, F>
where
    I: Iterator,
{
    inner: I,
    same_bucket: F,
    last: Option<Option<I::Item>>,
}

impl<I, F> DedupBy<I, F>
where
    I: Iterator,
{
    #[inline]
    pub(crate) fn new(inner: I, same_bucket: F) -> Self {
        Self { inner, same_bucket, last: None }
    }
}

impl<I, F> Iterator for DedupBy<I, F>
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { inner, last, same_bucket } = self;
        let last = last.get_or_insert_with(|| inner.next());
        let last_item = last.as_ref()?;
        let next = inner.find(|next_item| !(same_bucket)(next_item, last_item));
        crate::mem::replace(last, next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = matches!(self.last, Some(Some(_))).into();
        let max = self.inner.size_hint().1.map(|max| max + min);
        (min, max)
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, F> SourceIter for DedupBy<I, F>
where
    I: SourceIter + Iterator,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.inner) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, F> InPlaceIterable for DedupBy<I, F>
where
    I: InPlaceIterable,
    F: FnMut(&I::Item, &I::Item) -> bool,
{
}

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator that resolve to the same key.
///
/// This `struct` is created by  [`Iterator::dedup_by_key`].
/// See its documentation for more.
///
/// [`Iterator::dedup_by_key`]: Iterator::dedup_by_key
#[derive(Debug, Clone)]
pub struct DedupByKey<I, F, K>
where
    I: Iterator,
    F: FnMut(&I::Item) -> K,
{
    inner: I,
    key: F,
    last: Option<Option<I::Item>>,
}

impl<I, F, K> DedupByKey<I, F, K>
where
    I: Iterator,
    F: FnMut(&I::Item) -> K,
{
    #[inline]
    pub(crate) fn new(inner: I, key: F) -> Self {
        Self { inner, key, last: None }
    }
}

impl<I, F, K> Iterator for DedupByKey<I, F, K>
where
    I: Iterator,
    F: FnMut(&I::Item) -> K,
    K: PartialEq,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { inner, last, key } = self;
        let last = last.get_or_insert_with(|| inner.next());
        let last_item = last.as_ref()?;
        let next = inner.find(|next_item| key(next_item) != key(last_item));
        crate::mem::replace(last, next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = matches!(self.last, Some(Some(_))).into();
        let max = self.inner.size_hint().1.map(|max| max + min);
        (min, max)
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, F, K> SourceIter for DedupByKey<I, F, K>
where
    I: SourceIter + Iterator,
    F: FnMut(&I::Item) -> K,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.inner) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, F, K> InPlaceIterable for DedupByKey<I, F, K>
where
    I: InPlaceIterable,
    F: FnMut(&I::Item) -> K,
    K: PartialEq,
{
}

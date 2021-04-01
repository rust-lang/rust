use crate::mem::swap;

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator according to the [`PartialEq`] trait implementation.
///
/// This `struct` is created by [`Iterator::dedup`].
/// See its documentation for more.
///
/// [`Iterator::dedup`]: Iterator::dedup
#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
#[derive(Debug, Clone, Copy)]
pub struct Dedup<I, T> {
    inner: I,
    last: Option<T>,
}

impl<I, T> Dedup<I, T>
where
    I: Iterator<Item = T>,
{
    pub(crate) fn new(inner: I) -> Self {
        let mut inner = inner;
        Self { last: inner.next(), inner }
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<I, T> Iterator for Dedup<I, T>
where
    I: Iterator<Item = T>,
    T: PartialEq,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let last_item = self.last.as_ref()?;
        let mut next = loop {
            let curr = self.inner.next();
            if let Some(curr_item) = &curr {
                if last_item != curr_item {
                    break curr;
                }
            } else {
                break None;
            }
        };

        swap(&mut self.last, &mut next);
        next
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.size_hint().1)
    }
}

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator satisfying a given equality relation.
///
/// This `struct` is created by [`Iterator::dedup_by`].
/// See its documentation for more.
///
/// [`Iterator::dedup_by`]: Iterator::dedup_by
#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
#[derive(Debug, Clone, Copy)]
pub struct DedupBy<I, F, T> {
    inner: I,
    same_bucket: F,
    last: Option<T>,
}

impl<I, F, T> DedupBy<I, F, T>
where
    I: Iterator<Item = T>,
{
    pub(crate) fn new(inner: I, same_bucket: F) -> Self {
        let mut inner = inner;
        Self { last: inner.next(), inner, same_bucket }
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<I, F, T> Iterator for DedupBy<I, F, T>
where
    I: Iterator<Item = T>,
    F: FnMut(&T, &T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.last.is_none() {
            self.last = self.inner.next();
        }

        let last_item = self.last.as_ref()?;
        let mut next = loop {
            let curr = self.inner.next();
            if let Some(curr_item) = &curr {
                if !(self.same_bucket)(last_item, curr_item) {
                    break curr;
                }
            } else {
                break None;
            }
        };

        swap(&mut self.last, &mut next);
        next
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.size_hint().1)
    }
}

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator that resolve to the same key.
///
/// This `struct` is created by [`Iterator::dedup_by_key`].
/// See its documentation for more.
///
/// [`Iterator::dedup_by_key`]: Iterator::dedup_by_key
#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
#[derive(Debug, Clone, Copy)]
pub struct DedupByKey<I, F, T> {
    inner: I,
    key: F,
    last: Option<T>,
}

impl<I, F, T> DedupByKey<I, F, T>
where
    I: Iterator<Item = T>,
{
    pub(crate) fn new(inner: I, key: F) -> Self {
        let mut inner = inner;
        Self { last: inner.next(), inner, key }
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<I, F, K, T> Iterator for DedupByKey<I, F, T>
where
    I: Iterator<Item = T>,
    F: FnMut(&T) -> K,
    K: PartialEq,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.last.is_none() {
            self.last = self.inner.next();
        }

        let last_item = self.last.as_ref()?;
        let mut next = loop {
            let curr = self.inner.next();
            if let Some(curr_item) = &curr {
                if (self.key)(last_item) != (self.key)(curr_item) {
                    break curr;
                }
            } else {
                break None;
            }
        };

        swap(&mut self.last, &mut next);
        next
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.size_hint().1)
    }
}

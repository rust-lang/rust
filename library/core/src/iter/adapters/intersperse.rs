use core::fmt;
use core::iter::{Fuse, FusedIterator};

/// An iterator adapter that places a separator between all elements.
///
/// This `struct` is created by [`Iterator::intersperse`]. See its documentation
/// for more information.
#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
#[derive(Debug, Clone)]
pub struct Intersperse<I: Iterator>
where
    I::Item: Clone,
{
    separator: I::Item,
    next_item: Option<I::Item>,
    iter: Fuse<I>,
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I> FusedIterator for Intersperse<I>
where
    I: FusedIterator,
    I::Item: Clone,
{
}

impl<I: Iterator> Intersperse<I>
where
    I::Item: Clone,
{
    pub(in crate::iter) fn new(iter: I, separator: I::Item) -> Self {
        let mut iter = iter.fuse();
        Self { separator, next_item: iter.next(), iter }
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I> Iterator for Intersperse<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.next_item.take() {
            Some(v)
        } else {
            let next_item = self.iter.next();
            if next_item.is_some() {
                self.next_item = next_item;
                Some(self.separator.clone())
            } else {
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        intersperse_size_hint(&self.iter, self.next_item.is_some())
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let separator = self.separator;
        intersperse_fold(self.iter, init, f, move || separator.clone(), self.next_item)
    }
}

/// An iterator adapter that places a separator between all elements.
///
/// This `struct` is created by [`Iterator::intersperse_with`]. See its
/// documentation for more information.
#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
pub struct IntersperseWith<I, G>
where
    I: Iterator,
{
    separator: G,
    next_item: Option<I::Item>,
    iter: Fuse<I>,
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> FusedIterator for IntersperseWith<I, G>
where
    I: FusedIterator,
    G: FnMut() -> I::Item,
{
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> fmt::Debug for IntersperseWith<I, G>
where
    I: Iterator + fmt::Debug,
    I::Item: fmt::Debug,
    G: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntersperseWith")
            .field("separator", &self.separator)
            .field("iter", &self.iter)
            .field("next_item", &self.next_item)
            .finish()
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> Clone for IntersperseWith<I, G>
where
    I: Iterator + Clone,
    I::Item: Clone,
    G: Clone,
{
    fn clone(&self) -> Self {
        Self {
            separator: self.separator.clone(),
            iter: self.iter.clone(),
            next_item: self.next_item.clone(),
        }
    }
}

impl<I, G> IntersperseWith<I, G>
where
    I: Iterator,
    G: FnMut() -> I::Item,
{
    pub(in crate::iter) fn new(iter: I, separator: G) -> Self {
        let mut iter = iter.fuse();
        Self { separator, next_item: iter.next(), iter }
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> Iterator for IntersperseWith<I, G>
where
    I: Iterator,
    G: FnMut() -> I::Item,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.next_item.take() {
            Some(v)
        } else {
            let next_item = self.iter.next();
            if next_item.is_some() {
                self.next_item = next_item;
                Some((self.separator)())
            } else {
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        intersperse_size_hint(&self.iter, self.next_item.is_some())
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        intersperse_fold(self.iter, init, f, self.separator, self.next_item)
    }
}

fn intersperse_size_hint<I>(iter: &I, next_is_elem: bool) -> (usize, Option<usize>)
where
    I: Iterator,
{
    let (lo, hi) = iter.size_hint();
    (
        lo.saturating_add(next_is_elem as usize).saturating_add(lo),
        hi.map(|hi| hi.saturating_add(next_is_elem as usize).saturating_add(hi)),
    )
}

fn intersperse_fold<I, B, F, G>(
    iter: I,
    init: B,
    mut f: F,
    mut separator: G,
    mut next_item: Option<I::Item>,
) -> B
where
    I: Iterator,
    F: FnMut(B, I::Item) -> B,
    G: FnMut() -> I::Item,
{
    let mut accum = init;

    if let Some(x) = next_item.take() {
        accum = f(accum, x);
    }

    iter.fold(accum, |mut accum, x| {
        accum = f(accum, separator());
        accum = f(accum, x);
        accum
    })
}

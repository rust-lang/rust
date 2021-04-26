use crate::{
    iter::{InPlaceIterable, SourceIter},
    mem::swap,
};

/// A wrapper type around a key function.
///
/// This struct acts like a function which given a key function returns true
/// if and only if both arguments evaluate to the same key.
///
/// This `struct` is created by [`Iterator::dedup_by_key`].
/// See its documentation for more.
///
/// [`Iterator::dedup_by_key`]: Iterator::dedup_by_key
#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
#[derive(Debug, Clone, Copy)]
pub struct ByKey<F> {
    key: F,
}

impl<F> ByKey<F> {
    pub(crate) fn new(key: F) -> Self {
        Self { key }
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<F, T, K> FnOnce<(&T, &T)> for ByKey<F>
where
    F: FnMut(&T) -> K,
    K: PartialEq,
{
    type Output = bool;

    extern "rust-call" fn call_once(mut self, args: (&T, &T)) -> Self::Output {
        (self.key)(args.0) == (self.key)(args.1)
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<F, T, K> FnMut<(&T, &T)> for ByKey<F>
where
    F: FnMut(&T) -> K,
    K: PartialEq,
{
    extern "rust-call" fn call_mut(&mut self, args: (&T, &T)) -> Self::Output {
        (self.key)(args.0) == (self.key)(args.1)
    }
}

/// A zero-sized type for checking partial equality.
///
/// This struct acts exactly like the function [`PartialEq::eq`], but its
/// type is always known during compile time.
///
/// This `struct` is created by [`Iterator::dedup`].
/// See its documentation for more.
///
/// [`Iterator::dedup`]: Iterator::dedup
#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
#[derive(Debug, Clone, Copy)]
pub struct ByPartialEq;

impl ByPartialEq {
    pub(crate) fn new() -> Self {
        Self
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<T: PartialEq> FnOnce<(&T, &T)> for ByPartialEq {
    type Output = bool;

    extern "rust-call" fn call_once(self, args: (&T, &T)) -> Self::Output {
        args.0 == args.1
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<T: PartialEq> FnMut<(&T, &T)> for ByPartialEq {
    extern "rust-call" fn call_mut(&mut self, args: (&T, &T)) -> Self::Output {
        args.0 == args.1
    }
}

/// An iterator that removes all but the first of consecutive elements in a
/// given iterator satisfying a given equality relation.
///
/// This `struct` is created by [`Iterator::dedup`], [`Iterator::dedup_by`]
/// and [`Iterator::dedup_by_key`]. See its documentation for more.
///
/// [`Iterator::dedup`]: Iterator::dedup
/// [`Iterator::dedup_by`]: Iterator::dedup_by
/// [`Iterator::dedup_by_key`]: Iterator::dedup_by_key
#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
#[derive(Debug, Clone, Copy)]
pub struct Dedup<I, F, T> {
    inner: I,
    same_bucket: F,
    last: Option<T>,
}

impl<I, F, T> Dedup<I, F, T>
where
    I: Iterator<Item = T>,
{
    pub(crate) fn new(inner: I, same_bucket: F) -> Self {
        let mut inner = inner;
        Self { last: inner.next(), inner, same_bucket }
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
impl<I, F, T> Iterator for Dedup<I, F, T>
where
    I: Iterator<Item = T>,
    F: FnMut(&T, &T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
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
        let min = self.last.as_ref().map(|_| 1).unwrap_or(0);
        let max = self.inner.size_hint().1;
        (min, max)
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
unsafe impl<S, I, F, T> SourceIter for Dedup<I, F, T>
where
    S: Iterator,
    I: Iterator + SourceIter<Source = S>,
{
    type Source = S;

    unsafe fn as_inner(&mut self) -> &mut Self::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.inner) }
    }
}

#[unstable(feature = "iter_dedup", reason = "recently added", issue = "83748")]
unsafe impl<I, F, T> InPlaceIterable for Dedup<I, F, T>
where
    I: InPlaceIterable<Item = T>,
    F: FnMut(&T, &T) -> bool,
{
}

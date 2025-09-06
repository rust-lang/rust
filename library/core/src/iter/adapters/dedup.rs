trait DedupPredicate<T> {
    fn eq(&mut self, a: &T, b: &T) -> bool;
}

impl<T, F: FnMut(&T, &T) -> bool> DedupPredicate<T> for F {
    fn eq(&mut self, a: &T, b: &T) -> bool {
        self(a, b)
    }
}

#[unstable(feature = "iter_dedup", issue = "83747")]
#[doc(hidden)]
#[derive(Debug)]
pub struct DedupEq;

impl<T: PartialEq> DedupPredicate<T> for DedupEq {
    fn eq(&mut self, a: &T, b: &T) -> bool {
        a == b
    }
}

#[unstable(feature = "iter_dedup", issue = "83747")]
#[doc(hidden)]
#[derive(Debug)]
pub struct DedupKey<F>(pub F);

impl<T, K: PartialEq, F: Fn(&T) -> K> DedupPredicate<T> for DedupKey<F> {
    fn eq(&mut self, a: &T, b: &T) -> bool {
        (self.0)(a) == (self.0)(b)
    }
}

/// An iterator to deduplicate adjacent items in another iterator.
///
/// This `struct` is created by the [`dedup`], [`dedup_by`], and
/// [`dedup_by_key`] methods on [`Iterator`]. See their documentation for more.
///
/// [`dedup`]: Iterator::dedup
/// [`dedup_by`]: Iterator::dedup_by
/// [`dedup_by_key`]: Iterator::dedup_by_key
#[unstable(feature = "iter_dedup", issue = "83747")]
#[derive(Debug)]
pub struct Dedup<I: Iterator, F> {
    inner: I,
    f: F,
    last: Option<I::Item>,
}

impl<I: Iterator, F> Dedup<I, F> {
    pub(in crate::iter) fn new(mut it: I, f: F) -> Self {
        let first = it.next();
        Self { inner: it, f, last: first }
    }
}

#[unstable(feature = "iter_dedup", issue = "83747")]
impl<I, F> Iterator for Dedup<I, F>
where
    I: Iterator,
    I::Item: Clone,
    F: DedupPredicate<I::Item>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let last = self.last.as_ref()?;
        self.last = self.inner.find(|e| self.f.eq(e, last));
        return self.last.clone();
    }
}

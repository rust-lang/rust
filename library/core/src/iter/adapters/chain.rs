use crate::iter::{DoubleEndedIterator, FusedIterator, Iterator, TrustedLen};
use crate::num::NonZeroUsize;
use crate::ops::Try;

/// An iterator that links two iterators together, in a chain.
///
/// This `struct` is created by [`Iterator::chain`]. See its documentation
/// for more.
///
/// # Examples
///
/// ```
/// use std::iter::Chain;
/// use std::slice::Iter;
///
/// let a1 = [1, 2, 3];
/// let a2 = [4, 5, 6];
/// let iter: Chain<Iter<_>, Iter<_>> = a1.iter().chain(a2.iter());
/// ```
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chain<A, B> {
    // These are "fused" with `Option` so we don't need separate state to track which part is
    // already exhausted, and we may also get niche layout for `None`. We don't use the real `Fuse`
    // adapter because its specialization for `FusedIterator` unconditionally descends into the
    // iterator, and that could be expensive to keep revisiting stuff like nested chains. It also
    // hurts compiler performance to add more iterator layers to `Chain`.
    //
    // Only the "first" iterator is actually set `None` when exhausted, depending on whether you
    // iterate forward or backward. If you mix directions, then both sides may be `None`.
    a: Option<A>,
    b: Option<B>,
}
impl<A, B> Chain<A, B> {
    pub(in super::super) fn new(a: A, b: B) -> Chain<A, B> {
        Chain { a: Some(a), b: Some(b) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Chain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        and_then_or_clear(&mut self.a, Iterator::next).or_else(|| self.b.as_mut()?.next())
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn count(self) -> usize {
        let a_count = match self.a {
            Some(a) => a.count(),
            None => 0,
        };
        let b_count = match self.b {
            Some(b) => b.count(),
            None => 0,
        };
        a_count + b_count
    }

    fn try_fold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if let Some(ref mut a) = self.a {
            acc = a.try_fold(acc, &mut f)?;
            self.a = None;
        }
        if let Some(ref mut b) = self.b {
            acc = b.try_fold(acc, f)?;
            // we don't fuse the second iterator
        }
        try { acc }
    }

    fn fold<Acc, F>(self, mut acc: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(a) = self.a {
            acc = a.fold(acc, &mut f);
        }
        if let Some(b) = self.b {
            acc = b.fold(acc, f);
        }
        acc
    }

    #[inline]
    fn advance_by(&mut self, mut n: usize) -> Result<(), NonZeroUsize> {
        if let Some(ref mut a) = self.a {
            n = match a.advance_by(n) {
                Ok(()) => return Ok(()),
                Err(k) => k.get(),
            };
            self.a = None;
        }

        if let Some(ref mut b) = self.b {
            return b.advance_by(n);
            // we don't fuse the second iterator
        }

        NonZeroUsize::new(n).map_or(Ok(()), Err)
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if let Some(ref mut a) = self.a {
            n = match a.advance_by(n) {
                Ok(()) => match a.next() {
                    None => 0,
                    x => return x,
                },
                Err(k) => k.get(),
            };

            self.a = None;
        }

        self.b.as_mut()?.nth(n)
    }

    #[inline]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        and_then_or_clear(&mut self.a, |a| a.find(&mut predicate))
            .or_else(|| self.b.as_mut()?.find(predicate))
    }

    #[inline]
    fn last(self) -> Option<A::Item> {
        // Must exhaust a before b.
        let a_last = self.a.and_then(Iterator::last);
        let b_last = self.b.and_then(Iterator::last);
        b_last.or(a_last)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Chain { a: Some(a), b: Some(b) } => {
                let (a_lower, a_upper) = a.size_hint();
                let (b_lower, b_upper) = b.size_hint();

                let lower = a_lower.saturating_add(b_lower);

                let upper = match (a_upper, b_upper) {
                    (Some(x), Some(y)) => x.checked_add(y),
                    _ => None,
                };

                (lower, upper)
            }
            Chain { a: Some(a), b: None } => a.size_hint(),
            Chain { a: None, b: Some(b) } => b.size_hint(),
            Chain { a: None, b: None } => (0, Some(0)),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Chain<A, B>
where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item = A::Item>,
{
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        and_then_or_clear(&mut self.b, |b| b.next_back()).or_else(|| self.a.as_mut()?.next_back())
    }

    #[inline]
    fn advance_back_by(&mut self, mut n: usize) -> Result<(), NonZeroUsize> {
        if let Some(ref mut b) = self.b {
            n = match b.advance_back_by(n) {
                Ok(()) => return Ok(()),
                Err(k) => k.get(),
            };
            self.b = None;
        }

        if let Some(ref mut a) = self.a {
            return a.advance_back_by(n);
            // we don't fuse the second iterator
        }

        NonZeroUsize::new(n).map_or(Ok(()), Err)
    }

    #[inline]
    fn nth_back(&mut self, mut n: usize) -> Option<Self::Item> {
        if let Some(ref mut b) = self.b {
            n = match b.advance_back_by(n) {
                Ok(()) => match b.next_back() {
                    None => 0,
                    x => return x,
                },
                Err(k) => k.get(),
            };

            self.b = None;
        }

        self.a.as_mut()?.nth_back(n)
    }

    #[inline]
    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        and_then_or_clear(&mut self.b, |b| b.rfind(&mut predicate))
            .or_else(|| self.a.as_mut()?.rfind(predicate))
    }

    fn try_rfold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if let Some(ref mut b) = self.b {
            acc = b.try_rfold(acc, &mut f)?;
            self.b = None;
        }
        if let Some(ref mut a) = self.a {
            acc = a.try_rfold(acc, f)?;
            // we don't fuse the second iterator
        }
        try { acc }
    }

    fn rfold<Acc, F>(self, mut acc: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(b) = self.b {
            acc = b.rfold(acc, &mut f);
        }
        if let Some(a) = self.a {
            acc = a.rfold(acc, f);
        }
        acc
    }
}

// Note: *both* must be fused to handle double-ended iterators.
#[stable(feature = "fused", since = "1.26.0")]
impl<A, B> FusedIterator for Chain<A, B>
where
    A: FusedIterator,
    B: FusedIterator<Item = A::Item>,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A, B> TrustedLen for Chain<A, B>
where
    A: TrustedLen,
    B: TrustedLen<Item = A::Item>,
{
}

#[stable(feature = "default_iters", since = "CURRENT_RUSTC_VERSION")]
impl<A: Default, B: Default> Default for Chain<A, B> {
    /// Creates a `Chain` from the default values for `A` and `B`.
    ///
    /// ```
    /// # use core::iter::Chain;
    /// # use core::slice;
    /// # use std::collections::{btree_set, BTreeSet};
    /// # use std::mem;
    /// struct Foo<'a>(Chain<slice::Iter<'a, u8>, btree_set::Iter<'a, u8>>);
    ///
    /// let set = BTreeSet::<u8>::new();
    /// let slice: &[u8] = &[];
    /// let mut foo = Foo(slice.iter().chain(set.iter()));
    ///
    /// // take requires `Default`
    /// let _: Chain<_, _> = mem::take(&mut foo.0);
    fn default() -> Self {
        Chain::new(Default::default(), Default::default())
    }
}

#[inline]
fn and_then_or_clear<T, U>(opt: &mut Option<T>, f: impl FnOnce(&mut T) -> Option<U>) -> Option<U> {
    let x = f(opt.as_mut()?);
    if x.is_none() {
        *opt = None;
    }
    x
}

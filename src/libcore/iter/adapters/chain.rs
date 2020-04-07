use crate::iter::{DoubleEndedIterator, FusedIterator, Iterator, TrustedLen};
use crate::ops::Try;
use crate::usize;

/// An iterator that links two iterators together, in a chain.
///
/// This `struct` is created by the [`chain`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`chain`]: trait.Iterator.html#method.chain
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chain<A, B> {
    // These are "fused" with `Option` so we don't need separate state to track which part is
    // already exhausted, and we may also get niche layout for `None`. We don't use the real `Fuse`
    // adapter because its specialization for `FusedIterator` unconditionally descends into the
    // iterator, and that could be expensive to keep revisiting stuff like nested chains. It also
    // hurts compiler performance to add more iterator layers to `Chain`.
    a: Option<A>,
    b: Option<B>,
}
impl<A, B> Chain<A, B> {
    pub(in super::super) fn new(a: A, b: B) -> Chain<A, B> {
        Chain { a: Some(a), b: Some(b) }
    }
}

/// Fuse the iterator if the expression is `None`.
macro_rules! fuse {
    ($self:ident . $iter:ident . $($call:tt)+) => {
        match $self.$iter {
            Some(ref mut iter) => match iter.$($call)+ {
                None => {
                    $self.$iter = None;
                    None
                }
                item => item,
            },
            None => None,
        }
    };
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
        match fuse!(self.a.next()) {
            None => fuse!(self.b.next()),
            item => item,
        }
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
        R: Try<Ok = Acc>,
    {
        if let Some(ref mut a) = self.a {
            acc = a.try_fold(acc, &mut f)?;
            self.a = None;
        }
        if let Some(ref mut b) = self.b {
            acc = b.try_fold(acc, f)?;
            self.b = None;
        }
        Try::from_ok(acc)
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
    fn nth(&mut self, mut n: usize) -> Option<A::Item> {
        if let Some(ref mut a) = self.a {
            while let Some(x) = a.next() {
                if n == 0 {
                    return Some(x);
                }
                n -= 1;
            }
            self.a = None;
        }
        fuse!(self.b.nth(n))
    }

    #[inline]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        match fuse!(self.a.find(&mut predicate)) {
            None => fuse!(self.b.find(predicate)),
            item => item,
        }
    }

    #[inline]
    fn last(self) -> Option<A::Item> {
        // Must exhaust a before b.
        let a_last = match self.a {
            Some(a) => a.last(),
            None => None,
        };
        let b_last = match self.b {
            Some(b) => b.last(),
            None => None,
        };
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
        match fuse!(self.b.next_back()) {
            None => fuse!(self.a.next_back()),
            item => item,
        }
    }

    #[inline]
    fn nth_back(&mut self, mut n: usize) -> Option<A::Item> {
        if let Some(ref mut b) = self.b {
            while let Some(x) = b.next_back() {
                if n == 0 {
                    return Some(x);
                }
                n -= 1;
            }
            self.b = None;
        }
        fuse!(self.a.nth_back(n))
    }

    #[inline]
    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        match fuse!(self.b.rfind(&mut predicate)) {
            None => fuse!(self.a.rfind(predicate)),
            item => item,
        }
    }

    fn try_rfold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if let Some(ref mut b) = self.b {
            acc = b.try_rfold(acc, &mut f)?;
            self.b = None;
        }
        if let Some(ref mut a) = self.a {
            acc = a.try_rfold(acc, f)?;
            self.a = None;
        }
        Try::from_ok(acc)
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
// Now that we "fuse" both sides, we *could* implement this unconditionally,
// but we should be cautious about committing to that in the public API.
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

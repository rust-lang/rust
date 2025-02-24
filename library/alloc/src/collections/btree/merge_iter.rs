use core::cmp::Ordering;
use core::fmt::{self, Debug};
use core::iter::FusedIterator;

/// Core of an iterator that merges the output of two strictly ascending iterators,
/// for instance a union or a symmetric difference.
pub(super) struct MergeIterInner<I: Iterator> {
    a: I,
    b: I,
    peeked: Option<Peeked<I>>,
}

/// Benchmarks faster than wrapping both iterators in a Peekable,
/// probably because we can afford to impose a FusedIterator bound.
#[derive(Clone, Debug)]
enum Peeked<I: Iterator> {
    A(I::Item),
    B(I::Item),
}

impl<I: Iterator> Clone for MergeIterInner<I>
where
    I: Clone,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        Self { a: self.a.clone(), b: self.b.clone(), peeked: self.peeked.clone() }
    }
}

impl<I: Iterator> Debug for MergeIterInner<I>
where
    I: Debug,
    I::Item: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MergeIterInner").field(&self.a).field(&self.b).field(&self.peeked).finish()
    }
}

impl<I: Iterator> MergeIterInner<I> {
    /// Creates a new core for an iterator merging a pair of sources.
    pub(super) fn new(a: I, b: I) -> Self {
        MergeIterInner { a, b, peeked: None }
    }

    /// Returns the next pair of items stemming from the pair of sources
    /// being merged. If both returned options contain a value, that value
    /// is equal and occurs in both sources. If one of the returned options
    /// contains a value, that value doesn't occur in the other source (or
    /// the sources are not strictly ascending). If neither returned option
    /// contains a value, iteration has finished and subsequent calls will
    /// return the same empty pair.
    pub(super) fn nexts<Cmp: Fn(&I::Item, &I::Item) -> Ordering>(
        &mut self,
        cmp: Cmp,
    ) -> (Option<I::Item>, Option<I::Item>)
    where
        I: FusedIterator,
    {
        let mut a_next;
        let mut b_next;
        match self.peeked.take() {
            Some(Peeked::A(next)) => {
                a_next = Some(next);
                b_next = self.b.next();
            }
            Some(Peeked::B(next)) => {
                b_next = Some(next);
                a_next = self.a.next();
            }
            None => {
                a_next = self.a.next();
                b_next = self.b.next();
            }
        }
        if let (Some(a1), Some(b1)) = (&a_next, &b_next) {
            match cmp(a1, b1) {
                Ordering::Less => self.peeked = b_next.take().map(Peeked::B),
                Ordering::Greater => self.peeked = a_next.take().map(Peeked::A),
                Ordering::Equal => (),
            }
        }
        (a_next, b_next)
    }

    /// Returns a pair of upper bounds for the `size_hint` of the final iterator.
    pub(super) fn lens(&self) -> (usize, usize)
    where
        I: ExactSizeIterator,
    {
        match self.peeked {
            Some(Peeked::A(_)) => (1 + self.a.len(), self.b.len()),
            Some(Peeked::B(_)) => (self.a.len(), 1 + self.b.len()),
            _ => (self.a.len(), self.b.len()),
        }
    }
}

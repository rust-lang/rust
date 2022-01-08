use crate::fmt;
use crate::iter::{FusedIterator, Peekable};

/// An iterator that only accepts elements while `next` is `Some` `predicate` returns `true`.
///
/// This `struct` is created by the [`peekable_take_while`] method on [`Peekable`]. See its
/// documentation for more.
///
/// [`peekable_take_while`]: Peekable::peekable_take_while
/// [`Peekable`]: Peekable
#[unstable(feature = "peekable_map_while", issue = "none")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableTakeWhile<'iter, I: Iterator, P> {
    iter: &'iter mut Peekable<I>,
    flag: bool,
    predicate: P,
}

impl<'iter, I: Iterator, P> PeekableTakeWhile<'iter, I, P> {
    pub(in crate::iter) fn new(
        iter: &'iter mut Peekable<I>,
        predicate: P,
    ) -> PeekableTakeWhile<'iter, I, P> {
        PeekableTakeWhile { iter, flag: false, predicate }
    }
}

#[unstable(feature = "peekable_map_while", issue = "none")]
impl<'iter, I, P> fmt::Debug for PeekableTakeWhile<'iter, I, P>
where
    I: fmt::Debug + Iterator,
    <I as Iterator>::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeekableTakeWhile")
            .field("iter", &self.iter)
            .field("flag", &self.flag)
            .finish()
    }
}

#[unstable(feature = "peekable_map_while", issue = "none")]
impl<'iter, I: Iterator, P> Iterator for PeekableTakeWhile<'iter, I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.flag {
            None
        } else {
            let x = self.iter.peek()?;
            if (self.predicate)(x) {
                Some(self.iter.next().unwrap())
            } else {
                self.flag = true;
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.flag {
            (0, Some(0))
        } else {
            let (_, upper) = self.iter.size_hint();
            (0, upper) // can't know a lower bound, due to the predicate
        }
    }
}

#[unstable(feature = "peekable_map_while", issue = "none")]
impl<'iter, I, P> FusedIterator for PeekableTakeWhile<'iter, I, P>
where
    I: FusedIterator,
    P: FnMut(&I::Item) -> bool,
{
}

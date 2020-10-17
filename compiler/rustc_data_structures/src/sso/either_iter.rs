use std::fmt;
use std::iter::ExactSizeIterator;
use std::iter::FusedIterator;
use std::iter::Iterator;

/// Iterator which may contain instance of
/// one of two specific implementations.
///
/// Note: For most methods providing custom
///       implementation may margianlly
///       improve performance by avoiding
///       doing Left/Right match on every step
///       and doing it only once instead.
#[derive(Clone)]
pub enum EitherIter<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Iterator for EitherIter<L, R>
where
    L: Iterator,
    R: Iterator<Item = L::Item>,
{
    type Item = L::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EitherIter::Left(l) => l.next(),
            EitherIter::Right(r) => r.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            EitherIter::Left(l) => l.size_hint(),
            EitherIter::Right(r) => r.size_hint(),
        }
    }
}

impl<L, R> ExactSizeIterator for EitherIter<L, R>
where
    L: ExactSizeIterator,
    R: ExactSizeIterator,
    EitherIter<L, R>: Iterator,
{
    fn len(&self) -> usize {
        match self {
            EitherIter::Left(l) => l.len(),
            EitherIter::Right(r) => r.len(),
        }
    }
}

impl<L, R> FusedIterator for EitherIter<L, R>
where
    L: FusedIterator,
    R: FusedIterator,
    EitherIter<L, R>: Iterator,
{
}

impl<L, R> fmt::Debug for EitherIter<L, R>
where
    L: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EitherIter::Left(l) => l.fmt(f),
            EitherIter::Right(r) => r.fmt(f),
        }
    }
}

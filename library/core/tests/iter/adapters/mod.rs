mod chain;
mod cloned;
mod copied;
mod cycle;
mod enumerate;
mod filter;
mod filter_map;
mod flat_map;
mod flatten;
mod fuse;
mod inspect;
mod intersperse;
mod map;
mod peekable;
mod scan;
mod skip;
mod skip_while;
mod step_by;
mod take;
mod take_while;
mod zip;

use core::cell::Cell;

/// An iterator that panics whenever `next` or next_back` is called
/// after `None` has already been returned. This does not violate
/// `Iterator`'s contract. Used to test that iterator adapters don't
/// poll their inner iterators after exhausting them.
pub struct NonFused<I> {
    iter: I,
    done: bool,
}

impl<I> NonFused<I> {
    pub fn new(iter: I) -> Self {
        Self { iter, done: false }
    }
}

impl<I> Iterator for NonFused<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(!self.done, "this iterator has already returned None");
        self.iter.next().or_else(|| {
            self.done = true;
            None
        })
    }
}

impl<I> DoubleEndedIterator for NonFused<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        assert!(!self.done, "this iterator has already returned None");
        self.iter.next_back().or_else(|| {
            self.done = true;
            None
        })
    }
}

/// An iterator wrapper that panics whenever `next` or `next_back` is called
/// after `None` has been returned.
pub struct Unfuse<I> {
    iter: I,
    exhausted: bool,
}

impl<I> Unfuse<I> {
    pub fn new<T>(iter: T) -> Self
    where
        T: IntoIterator<IntoIter = I>,
    {
        Self { iter: iter.into_iter(), exhausted: false }
    }
}

impl<I> Iterator for Unfuse<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(!self.exhausted);
        let next = self.iter.next();
        self.exhausted = next.is_none();
        next
    }
}

impl<I> DoubleEndedIterator for Unfuse<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        assert!(!self.exhausted);
        let next = self.iter.next_back();
        self.exhausted = next.is_none();
        next
    }
}

pub struct Toggle {
    is_empty: bool,
}

impl Iterator for Toggle {
    type Item = ();

    // alternates between `None` and `Some(())`
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty {
            self.is_empty = false;
            None
        } else {
            self.is_empty = true;
            Some(())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_empty { (0, Some(0)) } else { (1, Some(1)) }
    }
}

impl DoubleEndedIterator for Toggle {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

/// This is an iterator that follows the Iterator contract,
/// but it is not fused. After having returned None once, it will start
/// producing elements if .next() is called again.
pub struct CycleIter<'a, T> {
    index: usize,
    data: &'a [T],
}

impl<'a, T> CycleIter<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        Self { index: 0, data }
    }
}

impl<'a, T> Iterator for CycleIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let elt = self.data.get(self.index);
        self.index += 1;
        self.index %= 1 + self.data.len();
        elt
    }
}

#[derive(Debug)]
struct CountClone(Cell<i32>);

impl CountClone {
    pub fn new() -> Self {
        Self(Cell::new(0))
    }
}

impl PartialEq<i32> for CountClone {
    fn eq(&self, rhs: &i32) -> bool {
        self.0.get() == *rhs
    }
}

impl Clone for CountClone {
    fn clone(&self) -> Self {
        let ret = CountClone(self.0.clone());
        let n = self.0.get();
        self.0.set(n + 1);
        ret
    }
}

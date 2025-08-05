#[unstable(feature = "peekable_iterator", issue = "132973")]
/// Iterators which inherently support `peek()` without needing to be wrapped by a `Peekable`.
pub trait PeekableIterator: Iterator {
    /// executes the closure with an Option containing `None` if the iterator is exhausted or Some(&Self::Item)
    fn peek_with<T>(&mut self, func: impl for<'a> FnOnce(Option<&'a Self::Item>) -> T) -> T;

    /// executes the closure on the next element without advancing the iterator, or returns None if the iterator is exhausted.
    fn peek_map<T>(&mut self, func: impl for<'a> FnOnce(&'a Self::Item) -> T) -> Option<T> {
        self.peek_with(|x| x.map(|y| func(y)))
    }

    /// returns the `next()` element if a predicate holds true
    fn next_if(&mut self, func: impl FnOnce(&Self::Item) -> bool) -> Option<Self::Item> {
        self.peek_map(func).and_then(|_| self.next())
    }

    /// move forward and return the `next()` item if it is equal to the expected value
    fn next_if_eq<T>(&mut self, expected: &T) -> Option<Self::Item>
    where
        Self::Item: PartialEq<T>,
        T: ?Sized,
    {
        self.next_if(|x| x == expected)
    }
}

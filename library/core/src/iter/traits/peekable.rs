#[unstable(feature = "peekable_iterator", issue = "132973")]
/// Iterators which inherently support peeking without needing to be wrapped by a `Peekable`.
pub trait PeekableIterator: Iterator {
    /// Executes the closure with a reference to the `next()` value without advancing the iterator.
    fn peek_with<T>(&mut self, func: impl for<'a> FnOnce(Option<&'a Self::Item>) -> T) -> T;

    /// Executes the closure on the `next()` element without advancing the iterator, or returns `None` if the iterator is exhausted.
    fn peek_map<T>(&mut self, func: impl for<'a> FnOnce(&'a Self::Item) -> T) -> Option<T> {
        self.peek_with(|x| x.map(|y| func(y)))
    }

    /// Returns the `next()` element if the given predicate holds true.
    fn next_if(&mut self, func: impl FnOnce(&Self::Item) -> bool) -> Option<Self::Item> {
        self.peek_with(|x| if func(x) { self.next() } else { None })
    }

    /// Moves forward and return the `next()` item if it is equal to the expected value.
    fn next_if_eq<T>(&mut self, expected: &T) -> Option<Self::Item>
    where
        Self::Item: PartialEq<T>,
        T: ?Sized,
    {
        self.next_if(|x| x == expected)
    }
}

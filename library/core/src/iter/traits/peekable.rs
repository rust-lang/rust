#[unstable(feature = "peekable_iterator", issue = "132973")]
/// Iterators which inherently support `peek()` without needing to be wrapped by a `Peekable`.
pub trait PeekableIterator: Iterator {
    /// returns a reference to the `next()` value without advancing the iterator.
    /// Just like `next()`, if there is a value, it returns a reference to it in Some()
    /// if the iteration is finished, a `None` is returned
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// let xs = [1, 2, 3];
    /// let mut iter = xs.iter();
    ///
    /// // peek() allows us to check the future value
    /// assert_eq!(iter.peek(), Some(&&1));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// // peek() doesn't move the iterator forward
    /// assert_eq!(iter.peek(), Some(&&2));
    /// assert_eq!(iter.peek(), Some(&&2));
    ///
    /// ```
    fn peek(&mut self) -> Option<&Self::Item>;

    /// returns the `next()` element if a predicate holds true
    fn next_if(&mut self, func: impl FnOnce(&Self::Item) -> bool) -> Option<Self::Item> {
        let Some(item) = self.peek() else {
            return None;
        };

        if func(item) { self.next() } else { None }
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

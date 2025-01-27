use core::borrow::Borrow;

/// An iterator with a `peek()` that returns an optional reference to the next
/// element.
#[unstable(feature = "peekable_iterator", issue = "132973")]
pub trait PeekableIterator: Iterator {
    /// The type of the item being peeked.
    type PeekedItem<'b>: Borrow<Self::Item> + 'b
    where
        Self: 'b;

    /// Returns a reference to the next() value without advancing the iterator.
    ///
    /// Like [`next`], if there is a value, it is wrapped in a `Some(T)`.
    /// But if the iteration is over, `None` is returned.
    ///
    /// [`next`]: Iterator::next
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(peekable_iterator)]
    /// # use std::iter::PeekableIterator;
    ///
    /// let xs = [1, 2, 3];
    ///
    /// let mut iter = xs.iter();
    ///
    /// // peek() lets us see into the future
    /// assert_eq!(iter.peek(), Some(&1));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// assert_eq!(iter.next(), Some(&2));
    ///
    /// // The iterator does not advance even if we `peek` multiple times
    /// assert_eq!(iter.peek(), Some(&3));
    /// assert_eq!(iter.peek(), Some(&3));
    ///
    /// assert_eq!(iter.next(), Some(&3));
    ///
    /// // After the iterator is finished, so is `peek()`
    /// assert_eq!(iter.peek(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    fn peek(&self) -> Option<Self::PeekedItem<'_>>;

    /// Consumes and return the next value of this iterator if a condition is true.
    ///
    /// If `func` returns `true` for the next value of this iterator, consume and return it.
    /// Otherwise, return `None`.
    ///
    /// # Examples
    /// Consume a number if it's equal to 0.
    /// ```
    /// #![feature(peekable_iterator)]
    /// # use std::iter::PeekableIterator;
    ///
    /// let mut iter = 0..5;
    /// // The first item of the iterator is 0; consume it.
    /// assert_eq!(iter.next_if(|&x| x == 0), Some(0));
    /// // The next item returned is now 1, so `next_if` will return `None`.
    /// assert_eq!(iter.next_if(|&x| x == 0), None);
    /// // `next_if` saves the value of the next item if it was not equal to `expected`.
    /// assert_eq!(iter.next(), Some(1));
    /// ```
    ///
    /// Consume any number less than 10.
    /// ```
    /// #![feature(peekable_iterator)]
    /// # use std::iter::PeekableIterator;
    ///
    /// let mut iter = 1..20;
    /// // Consume all numbers less than 10
    /// while iter.next_if(|&x| x < 10).is_some() {}
    /// // The next value returned will be 10
    /// assert_eq!(iter.next(), Some(10));
    /// ```
    fn next_if(&mut self, func: impl FnOnce(&Self::Item) -> bool) -> Option<Self::Item> {
        match self.peek() {
            Some(matched) if func(matched.borrow()) => (),
            _ => return None,
        };
        self.next()
    }

    /// Consumes and return the next item if it is equal to `expected`.
    ///
    /// # Example
    /// Consume a number if it's equal to 0.
    /// ```
    /// #![feature(peekable_iterator)]
    /// # use std::iter::PeekableIterator;
    ///
    /// let mut iter = 0..5;
    /// // The first item of the iterator is 0; consume it.
    /// assert_eq!(iter.next_if_eq(&0), Some(0));
    /// // The next item returned is now 1, so `next_if` will return `None`.
    /// assert_eq!(iter.next_if_eq(&0), None);
    /// // `next_if_eq` saves the value of the next item if it was not equal to `expected`.
    /// assert_eq!(iter.next(), Some(1));
    /// ```
    fn next_if_eq<T>(&mut self, expected: &T) -> Option<Self::Item>
    where
        Self::Item: PartialEq<T>,
        T: ?Sized,
    {
        self.next_if(|next| next == expected)
    }
}

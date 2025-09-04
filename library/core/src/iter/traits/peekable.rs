#[unstable(feature = "peekable_iterator", issue = "132973")]
/// Iterators which inherently support peeking without needing to be wrapped by a `Peekable`.
pub trait PeekableIterator: Iterator {
    /// Executes the closure with a reference to the `next()` value without advancing the iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// #![feature(peekable_iterator)]
    /// use std::iter::PeekableIterator;
    ///
    /// let mut vals = [0, 1, 2].into_iter();
    ///
    /// assert_eq!(vals.peek_with(|x| x.copied()), Some(0));
    /// // element is not consumed
    /// assert_eq!(vals.next(), Some(0));
    ///
    /// // examine the pending element
    /// assert_eq!(vals.peek_with(|x| x), Some(&1));
    /// assert_eq!(vals.next(), Some(1));
    ///
    /// // determine if the iterator has an element without advancing
    /// assert_eq!(vals.peek_with(|x| x.is_some()), false);
    /// assert_eq!(vals.next(), Some(2));
    ///
    /// // exhausted iterator
    /// assert_eq!(vals.next(), None);
    /// assert_eq!(vals.peek_with(|x| x), None);
    /// ```
    fn peek_with<T>(&mut self, func: impl for<'a> FnOnce(Option<&'a Self::Item>) -> T) -> T;

    /// Returns the `next()` element if the given predicate holds true.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// #![feature(peekable_iterator)]
    /// use std::iter::PeekableIterator;
    /// fn parse_number(s: &str) -> u32 {
    ///     let mut c = s.chars();
    ///
    ///     let base = if c.next_if_eq(&"0").is_some() {
    ///         match c.next_if(|c| "oxb".contains(c)) {
    ///             Some("x") => 16,
    ///             Some("b") => 2,
    ///             _ => 8
    ///         }
    ///     } else {
    ///       10
    ///     }
    ///     
    ///     u32::from_str_radix(c.as_str(), base).unwrap()
    /// }
    ///
    /// assert_eq!(parse_number("055"), 45);
    /// assert_eq!(parse_number("0o42"), 34);
    /// assert_eq!(parse_number("0x11"), 17);
    /// ```
    ///
    fn next_if(&mut self, func: impl FnOnce(&Self::Item) -> bool) -> Option<Self::Item> {
        match self.peek_with(|x| x.map(|y| func(y))) {
            Some(true) => self.next(),
            _ => None,
        }
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

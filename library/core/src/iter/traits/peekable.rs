/// An iterator whose elements may be checked without advancing the iterator.
///
/// In general, many [`Iterator`]s are able to "peek" their next element,
/// showing what would be returned by a call to `next` without advancing their
/// internal state. If these iterators do not offer such functionality, it can
/// be manually added to an iterator using the [`peekable`] method.
///
/// In most cases, calling [`peekable`] on a [`PeekableIterator`] shouldn't
/// noticeably affect functionality, however, it's worth pointing out a few
/// differences between [`peekable`] and [`PeekableIterator`]:
///
/// * Stateful iterators like those using [`inspect`](Iterator::inspect) will
///   eagerly evaluate when peeked by a [`peekable`] wrapper, but may do so
///   lazily with a custom [`PeekableIterator`] implementation.
/// * The [`peekable`] wrapper will incur a small performance penalty for
///   [`next`] and [`next_back`], but [`PeekableIterator`] implementations
///   incur no such penalty.
/// * The [`peekable`] wrapper will return a reference to its item, whereas
///   [`PeekableIterator`] will return the item directly.
///
/// Note that this trait is a safe trait and as such does *not* and *cannot*
/// guarantee that the peeked value will be returned in a subsequent call to
/// [`next`], no matter how soon the two are called together. A common
/// example of this is interior mutability; if the interior state of the
/// iterator is mutated between a call to [`peek`] and [`next`], then the
/// values may differ.
///
/// [`peek`]: Self::peek
/// [`peekable`]: Iterator::peekable
/// [`Peekable`]: super::Peekable
/// [`next`]: Iterator::next
/// [`next_back`]: DoubleEndedIterator::next_back
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// #![feature(peekable_iterator)]
/// use std::iter::PeekableIterator;
///
/// // a range knows its current state exactly
/// let five = 0..5;
///
/// assert_eq!(Some(0), five.peek());
/// ```
#[unstable(feature = "peekable_iterator", issue = "none")]
pub trait PeekableIterator: Iterator {
    /// Returns a reference to the [`next`] value without advancing the iterator.
    ///
    /// [`next`]: Iterator::next
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(peekable_iterator)]
    /// use std::iter::PeekableIterator;
    ///
    /// // a finite range knows its current state exactly
    /// let five = 0..5;
    ///
    /// assert_eq!(Some(0), five.peek());
    /// ```
    #[unstable(feature = "peekable_iterator", issue = "none")]
    fn peek(&self) -> Option<Self::Item>;

    /// Returns `true` if the [`next`] value is `None`.
    ///
    /// [`next`]: Iterator::next
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(peekable_iterator)]
    /// use std::iter::PeekableIterator;
    ///
    /// let mut one_element = std::iter::once(0);
    /// assert!(one_element.has_next());
    ///
    /// assert_eq!(one_element.next(), Some(0));
    /// assert!(!one_element.has_next());
    ///
    /// assert_eq!(one_element.next(), None);
    /// ```
    #[inline]
    #[unstable(feature = "peekable_iterator", issue = "none")]
    fn has_next(&self) -> bool {
        self.peek().is_some()
    }
}

#[unstable(feature = "peekable_iterator", issue = "none")]
impl<I: PeekableIterator + ?Sized> PeekableIterator for &mut I {
    fn peek(&self) -> Option<I::Item> {
        (**self).peek()
    }
}

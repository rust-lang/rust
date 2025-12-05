/// An iterator that knows its exact length.
///
/// Many [`Iterator`]s don't know how many times they will iterate, but some do.
/// If an iterator knows how many times it can iterate, providing access to
/// that information can be useful. For example, if you want to iterate
/// backwards, a good start is to know where the end is.
///
/// When implementing an `ExactSizeIterator`, you must also implement
/// [`Iterator`]. When doing so, the implementation of [`Iterator::size_hint`]
/// *must* return the exact size of the iterator.
///
/// The [`len`] method has a default implementation, so you usually shouldn't
/// implement it. However, you may be able to provide a more performant
/// implementation than the default, so overriding it in this case makes sense.
///
/// Note that this trait is a safe trait and as such does *not* and *cannot*
/// guarantee that the returned length is correct. This means that `unsafe`
/// code **must not** rely on the correctness of [`Iterator::size_hint`]. The
/// unstable and unsafe [`TrustedLen`](super::marker::TrustedLen) trait gives
/// this additional guarantee.
///
/// [`len`]: ExactSizeIterator::len
///
/// # When *shouldn't* an adapter be `ExactSizeIterator`?
///
/// If an adapter makes an iterator *longer*, then it's usually incorrect for
/// that adapter to implement `ExactSizeIterator`.  The inner exact-sized
/// iterator might already be `usize::MAX`-long, and thus the length of the
/// longer adapted iterator would no longer be exactly representable in `usize`.
///
/// This is why [`Chain<A, B>`](crate::iter::Chain) isn't `ExactSizeIterator`,
/// even when `A` and `B` are both `ExactSizeIterator`.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // a finite range knows exactly how many times it will iterate
/// let five = 0..5;
///
/// assert_eq!(5, five.len());
/// ```
///
/// In the [module-level docs], we implemented an [`Iterator`], `Counter`.
/// Let's implement `ExactSizeIterator` for it as well:
///
/// [module-level docs]: crate::iter
///
/// ```
/// # struct Counter {
/// #     count: usize,
/// # }
/// # impl Counter {
/// #     fn new() -> Counter {
/// #         Counter { count: 0 }
/// #     }
/// # }
/// # impl Iterator for Counter {
/// #     type Item = usize;
/// #     fn next(&mut self) -> Option<Self::Item> {
/// #         self.count += 1;
/// #         if self.count < 6 {
/// #             Some(self.count)
/// #         } else {
/// #             None
/// #         }
/// #     }
/// # }
/// impl ExactSizeIterator for Counter {
///     // We can easily calculate the remaining number of iterations.
///     fn len(&self) -> usize {
///         5 - self.count
///     }
/// }
///
/// // And now we can use it!
///
/// let mut counter = Counter::new();
///
/// assert_eq!(5, counter.len());
/// let _ = counter.next();
/// assert_eq!(4, counter.len());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExactSizeIterator: Iterator {
    /// Returns the exact remaining length of the iterator.
    ///
    /// The implementation ensures that the iterator will return exactly `len()`
    /// more times a [`Some(T)`] value, before returning [`None`].
    /// This method has a default implementation, so you usually should not
    /// implement it directly. However, if you can provide a more efficient
    /// implementation, you can do so. See the [trait-level] docs for an
    /// example.
    ///
    /// This function has the same safety guarantees as the
    /// [`Iterator::size_hint`] function.
    ///
    /// [trait-level]: ExactSizeIterator
    /// [`Some(T)`]: Some
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // a finite range knows exactly how many times it will iterate
    /// let mut range = 0..5;
    ///
    /// assert_eq!(5, range.len());
    /// let _ = range.next();
    /// assert_eq!(4, range.len());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }

    /// Returns `true` if the iterator is empty.
    ///
    /// This method has a default implementation using
    /// [`ExactSizeIterator::len()`], so you don't need to implement it yourself.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(exact_size_is_empty)]
    ///
    /// let mut one_element = std::iter::once(0);
    /// assert!(!one_element.is_empty());
    ///
    /// assert_eq!(one_element.next(), Some(0));
    /// assert!(one_element.is_empty());
    ///
    /// assert_eq!(one_element.next(), None);
    /// ```
    #[inline]
    #[unstable(feature = "exact_size_is_empty", issue = "35428")]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized> ExactSizeIterator for &mut I {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

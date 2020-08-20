//! Allocation extensions for [`Iterator`].
//!
//! *[See also the Iterator trait][Iterator].*
#![unstable(feature = "iterator_join", issue = "75638")]

use crate::slice::Join;
use crate::vec::Vec;

/// Iterator extension traits that requires allocation.
#[unstable(feature = "iterator_join", issue = "75638")]
pub trait IteratorExt: Iterator {
    /// Flattens an iterator into a single value with the given separator in
    /// between.
    ///
    /// Combines `collect` with `join` to convert a sequence into a value
    /// separated with the specified separator.
    ///
    /// Allows `.join(sep)` instead of `.collect::<Vec<_>>().join(sep)`.
    ///
    /// ```
    /// #![feature(iterator_join)]
    /// use alloc::iter::IteratorExt;
    ///
    /// assert_eq!(["hello", "world"].iter().copied().join(" "), "hello world");
    /// assert_eq!([[1, 2], [3, 4]].iter().copied().join(&0), [1, 2, 0, 3, 4]);
    /// assert_eq!([[1, 2], [3, 4]].iter().copied().join(&[0, 0][..]), [1, 2, 0, 0, 3, 4]);
    /// ```
    #[inline]
    #[unstable(feature = "iterator_join", issue = "75638")]
    #[must_use = "if you really need to exhaust the iterator, consider `.for_each(drop)` instead"]
    fn join<Separator>(self, sep: Separator) -> <[Self::Item] as Join<Separator>>::Output
    where
        [Self::Item]: Join<Separator>,
        Self: Sized,
    {
        Join::join(self.collect::<Vec<Self::Item>>().as_slice(), sep)
    }
}

impl<T: Iterator + ?Sized> IteratorExt for T {}

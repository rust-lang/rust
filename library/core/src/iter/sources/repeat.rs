use crate::iter::{FusedIterator, TrustedLen};
use crate::num::NonZero;

/// Creates a new iterator that endlessly repeats a single element.
///
/// The `repeat()` function repeats a single value over and over again.
///
/// Infinite iterators like `repeat()` are often used with adapters like
/// [`Iterator::take()`], in order to make them finite.
///
/// Use [`str::repeat()`] instead of this function if you just want to repeat
/// a char/string `n` times.
///
/// If the element type of the iterator you need does not implement `Clone`,
/// or if you do not want to keep the repeated element in memory, you can
/// instead use the [`repeat_with()`] function.
///
/// [`repeat_with()`]: crate::iter::repeat_with
/// [`str::repeat()`]: ../../std/primitive.str.html#method.repeat
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // the number four 4ever:
/// let mut fours = iter::repeat(4);
///
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
///
/// // yup, still four
/// assert_eq!(Some(4), fours.next());
/// ```
///
/// Going finite with [`Iterator::take()`]:
///
/// ```
/// use std::iter;
///
/// // that last example was too many fours. Let's only have four fours.
/// let mut four_fours = iter::repeat(4).take(4);
///
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
///
/// // ... and now we're done
/// assert_eq!(None, four_fours.next());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "iter_repeat"]
pub fn repeat<T: Clone>(elt: T) -> Repeat<T> {
    Repeat { element: elt }
}

/// An iterator that repeats an element endlessly.
///
/// This `struct` is created by the [`repeat()`] function. See its documentation for more.
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Repeat<A> {
    element: A,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> Iterator for Repeat<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        Some(self.element.clone())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        // Advancing an infinite iterator of a single element is a no-op.
        let _ = n;
        Ok(())
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        let _ = n;
        Some(self.element.clone())
    }

    fn last(self) -> Option<A> {
        Some(self.element)
    }

    #[track_caller]
    fn count(self) -> usize {
        panic!("iterator is infinite");
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> DoubleEndedIterator for Repeat<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        Some(self.element.clone())
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        // Advancing an infinite iterator of a single element is a no-op.
        let _ = n;
        Ok(())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        let _ = n;
        Some(self.element.clone())
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Clone> FusedIterator for Repeat<A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: Clone> TrustedLen for Repeat<A> {}

use crate::iter::{FusedIterator, TrustedLen};
use crate::mem::ManuallyDrop;
use crate::num::NonZeroUsize;

/// Creates a new iterator that repeats a single element a given number of times.
///
/// The `repeat_n()` function repeats a single value exactly `n` times.
///
/// This is very similar to using [`repeat()`] with [`Iterator::take()`],
/// but there are two differences:
/// - `repeat_n()` can return the original value, rather than always cloning.
/// - `repeat_n()` produces an [`ExactSizeIterator`].
///
/// [`repeat()`]: crate::iter::repeat
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// #![feature(iter_repeat_n)]
/// use std::iter;
///
/// // four of the number four:
/// let mut four_fours = iter::repeat_n(4, 4);
///
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
///
/// // no more fours
/// assert_eq!(None, four_fours.next());
/// ```
///
/// For non-`Copy` types,
///
/// ```
/// #![feature(iter_repeat_n)]
/// use std::iter;
///
/// let v: Vec<i32> = Vec::with_capacity(123);
/// let mut it = iter::repeat_n(v, 5);
///
/// for i in 0..4 {
///     // It starts by cloning things
///     let cloned = it.next().unwrap();
///     assert_eq!(cloned.len(), 0);
///     assert_eq!(cloned.capacity(), 0);
/// }
///
/// // ... but the last item is the original one
/// let last = it.next().unwrap();
/// assert_eq!(last.len(), 0);
/// assert_eq!(last.capacity(), 123);
///
/// // ... and now we're done
/// assert_eq!(None, it.next());
/// ```
#[inline]
#[unstable(feature = "iter_repeat_n", issue = "104434")]
#[doc(hidden)] // waiting on ACP#120 to decide whether to expose publicly
pub fn repeat_n<T: Clone>(element: T, count: usize) -> RepeatN<T> {
    let mut element = ManuallyDrop::new(element);

    if count == 0 {
        // SAFETY: we definitely haven't dropped it yet, since we only just got
        // passed it in, and because the count is zero the instance we're about
        // to create won't drop it, so to avoid leaking we need to now.
        unsafe { ManuallyDrop::drop(&mut element) };
    }

    RepeatN { element, count }
}

/// An iterator that repeats an element an exact number of times.
///
/// This `struct` is created by the [`repeat_n()`] function.
/// See its documentation for more.
#[derive(Clone, Debug)]
#[unstable(feature = "iter_repeat_n", issue = "104434")]
#[doc(hidden)] // waiting on ACP#120 to decide whether to expose publicly
pub struct RepeatN<A> {
    count: usize,
    // Invariant: has been dropped iff count == 0.
    element: ManuallyDrop<A>,
}

impl<A> RepeatN<A> {
    /// If we haven't already dropped the element, return it in an option.
    ///
    /// Clears the count so it won't be dropped again later.
    #[inline]
    fn take_element(&mut self) -> Option<A> {
        if self.count > 0 {
            self.count = 0;
            // SAFETY: We just set count to zero so it won't be dropped again,
            // and it used to be non-zero so it hasn't already been dropped.
            unsafe { Some(ManuallyDrop::take(&mut self.element)) }
        } else {
            None
        }
    }
}

#[unstable(feature = "iter_repeat_n", issue = "104434")]
impl<A> Drop for RepeatN<A> {
    fn drop(&mut self) {
        self.take_element();
    }
}

#[unstable(feature = "iter_repeat_n", issue = "104434")]
impl<A: Clone> Iterator for RepeatN<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.count == 0 {
            return None;
        }

        self.count -= 1;
        Some(if self.count == 0 {
            // SAFETY: the check above ensured that the count used to be non-zero,
            // so element hasn't been dropped yet, and we just lowered the count to
            // zero so it won't be dropped later, and thus it's okay to take it here.
            unsafe { ManuallyDrop::take(&mut self.element) }
        } else {
            A::clone(&self.element)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn advance_by(&mut self, skip: usize) -> Result<(), NonZeroUsize> {
        let len = self.count;

        if skip >= len {
            self.take_element();
        }

        if skip > len {
            // SAFETY: we just checked that the difference is positive
            Err(unsafe { NonZeroUsize::new_unchecked(skip - len) })
        } else {
            self.count = len - skip;
            Ok(())
        }
    }

    #[inline]
    fn last(mut self) -> Option<A> {
        self.take_element()
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

#[unstable(feature = "iter_repeat_n", issue = "104434")]
impl<A: Clone> ExactSizeIterator for RepeatN<A> {
    fn len(&self) -> usize {
        self.count
    }
}

#[unstable(feature = "iter_repeat_n", issue = "104434")]
impl<A: Clone> DoubleEndedIterator for RepeatN<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.next()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.advance_by(n)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.nth(n)
    }
}

#[unstable(feature = "iter_repeat_n", issue = "104434")]
impl<A: Clone> FusedIterator for RepeatN<A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: Clone> TrustedLen for RepeatN<A> {}

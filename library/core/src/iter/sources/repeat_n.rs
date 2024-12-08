use crate::fmt;
use crate::iter::{FusedIterator, TrustedLen, UncheckedIterator};
use crate::mem::{self, MaybeUninit};
use crate::num::NonZero;

/// Creates a new iterator that repeats a single element a given number of times.
///
/// The `repeat_n()` function repeats a single value exactly `n` times.
///
/// This is very similar to using [`repeat()`] with [`Iterator::take()`],
/// but `repeat_n()` can return the original value, rather than always cloning.
///
/// [`repeat()`]: crate::iter::repeat
///
/// # Examples
///
/// Basic usage:
///
/// ```
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
#[stable(feature = "iter_repeat_n", since = "1.82.0")]
pub fn repeat_n<T: Clone>(element: T, count: usize) -> RepeatN<T> {
    let element = if count == 0 {
        // `element` gets dropped eagerly.
        MaybeUninit::uninit()
    } else {
        MaybeUninit::new(element)
    };

    RepeatN { element, count }
}

/// An iterator that repeats an element an exact number of times.
///
/// This `struct` is created by the [`repeat_n()`] function.
/// See its documentation for more.
#[stable(feature = "iter_repeat_n", since = "1.82.0")]
pub struct RepeatN<A> {
    count: usize,
    // Invariant: uninit iff count == 0.
    element: MaybeUninit<A>,
}

impl<A> RepeatN<A> {
    /// Returns the element if it hasn't been dropped already.
    fn element_ref(&self) -> Option<&A> {
        if self.count > 0 {
            // SAFETY: The count is non-zero, so it must be initialized.
            Some(unsafe { self.element.assume_init_ref() })
        } else {
            None
        }
    }
    /// If we haven't already dropped the element, return it in an option.
    ///
    /// Clears the count so it won't be dropped again later.
    #[inline]
    fn take_element(&mut self) -> Option<A> {
        if self.count > 0 {
            self.count = 0;
            let element = mem::replace(&mut self.element, MaybeUninit::uninit());
            // SAFETY: We just set count to zero so it won't be dropped again,
            // and it used to be non-zero so it hasn't already been dropped.
            unsafe { Some(element.assume_init()) }
        } else {
            None
        }
    }
}

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: Clone> Clone for RepeatN<A> {
    fn clone(&self) -> RepeatN<A> {
        RepeatN {
            count: self.count,
            element: self.element_ref().cloned().map_or_else(MaybeUninit::uninit, MaybeUninit::new),
        }
    }
}

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: fmt::Debug> fmt::Debug for RepeatN<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RepeatN")
            .field("count", &self.count)
            .field("element", &self.element_ref())
            .finish()
    }
}

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A> Drop for RepeatN<A> {
    fn drop(&mut self) {
        self.take_element();
    }
}

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: Clone> Iterator for RepeatN<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.count > 0 {
            // SAFETY: Just checked it's not empty
            unsafe { Some(self.next_unchecked()) }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn advance_by(&mut self, skip: usize) -> Result<(), NonZero<usize>> {
        let len = self.count;

        if skip >= len {
            self.take_element();
        }

        if skip > len {
            // SAFETY: we just checked that the difference is positive
            Err(unsafe { NonZero::new_unchecked(skip - len) })
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

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: Clone> ExactSizeIterator for RepeatN<A> {
    fn len(&self) -> usize {
        self.count
    }
}

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: Clone> DoubleEndedIterator for RepeatN<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.next()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.advance_by(n)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.nth(n)
    }
}

#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: Clone> FusedIterator for RepeatN<A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: Clone> TrustedLen for RepeatN<A> {}
#[stable(feature = "iter_repeat_n", since = "1.82.0")]
impl<A: Clone> UncheckedIterator for RepeatN<A> {
    #[inline]
    unsafe fn next_unchecked(&mut self) -> Self::Item {
        // SAFETY: The caller promised the iterator isn't empty
        self.count = unsafe { self.count.unchecked_sub(1) };
        if self.count == 0 {
            // SAFETY: the check above ensured that the count used to be non-zero,
            // so element hasn't been dropped yet, and we just lowered the count to
            // zero so it won't be dropped later, and thus it's okay to take it here.
            unsafe { mem::replace(&mut self.element, MaybeUninit::uninit()).assume_init() }
        } else {
            // SAFETY: the count is non-zero, so it must have not been dropped yet.
            let element = unsafe { self.element.assume_init_ref() };
            A::clone(element)
        }
    }
}

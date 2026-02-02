use core::alloc::Allocator;

use crate::alloc::Global;
use crate::collections::vec_deque::Drain;
use crate::vec::Vec;

/// A splicing iterator for `VecDeque`.
///
/// This struct is created by [`VecDeque::splice()`][super::VecDeque::splice].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// # #![feature(deque_extend_front)]
/// # use std::collections::VecDeque;
///
/// let mut v = VecDeque::from(vec![0, 1, 2]);
/// let new = [7, 8];
/// let iter: std::collections::vec_deque::Splice<'_, _> = v.splice(1.., new);
/// ```
#[unstable(feature = "deque_extend_front", issue = "146975")]
#[derive(Debug)]
pub struct Splice<
    'a,
    I: Iterator + 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + 'a = Global,
> {
    pub(super) drain: Drain<'a, I::Item, A>,
    pub(super) replace_with: I,
}

#[unstable(feature = "deque_extend_front", issue = "146975")]
impl<I: Iterator, A: Allocator> Iterator for Splice<'_, I, A> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

#[unstable(feature = "deque_extend_front", issue = "146975")]
impl<I: Iterator, A: Allocator> DoubleEndedIterator for Splice<'_, I, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

#[unstable(feature = "deque_extend_front", issue = "146975")]
impl<I: Iterator, A: Allocator> ExactSizeIterator for Splice<'_, I, A> {}

// See also: [`crate::vec::Splice`].
#[unstable(feature = "deque_extend_front", issue = "146975")]
impl<I: Iterator, A: Allocator> Drop for Splice<'_, I, A> {
    fn drop(&mut self) {
        // This will set drain.remaining to 0, so its drop won't try to read deallocated memory on
        // drop.
        self.drain.by_ref().for_each(drop);

        // At this point draining is done and the only remaining tasks are splicing
        // and moving things into the final place.

        unsafe {
            let tail_len = self.drain.tail_len; // #elements behind the drain

            if tail_len == 0 {
                self.drain.deque.as_mut().extend(self.replace_with.by_ref());
                return;
            }

            // First fill the range left by drain().
            if !self.drain.fill(&mut self.replace_with) {
                return;
            }

            // There may be more elements. Use the lower bound as an estimate.
            // FIXME: Is the upper bound a better guess? Or something else?
            let (lower_bound, _upper_bound) = self.replace_with.size_hint();
            if lower_bound > 0 {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return;
                }
            }

            // Collect any remaining elements.
            // This is a zero-length vector which does not allocate if `lower_bound` was exact.
            let mut collected = self.replace_with.by_ref().collect::<Vec<I::Item>>().into_iter();
            // Now we have an exact count.
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                let filled = self.drain.fill(&mut collected);
                debug_assert!(filled);
                debug_assert_eq!(collected.len(), 0);
            }
        }
        // Let `Drain::drop` move the tail back if necessary and restore `deque.len`.
    }
}

/// Private helper methods for `Splice::drop`
impl<T, A: Allocator> Drain<'_, T, A> {
    /// The range from `self.deque.len` to `self.deque.len + self.drain_len` contains elements that
    /// have been moved out.
    /// Fill that range as much as possible with new elements from the `replace_with` iterator.
    /// Returns `true` if we filled the entire range. (`replace_with.next()` didn’t return `None`.)
    ///
    /// # Safety
    ///
    /// self.deque must be valid. self.deque.len and self.deque.len + self.drain_len must be less
    /// than twice the deque's capacity.
    unsafe fn fill<I: Iterator<Item = T>>(&mut self, replace_with: &mut I) -> bool {
        let deque = unsafe { self.deque.as_mut() };
        let range_start = deque.len;
        let range_end = range_start + self.drain_len;

        for idx in range_start..range_end {
            if let Some(new_item) = replace_with.next() {
                let index = deque.to_physical_idx(idx);
                unsafe { deque.buffer_write(index, new_item) };
                deque.len += 1;
                self.drain_len -= 1;
            } else {
                return false;
            }
        }
        true
    }

    /// Makes room for inserting more elements before the tail.
    ///
    /// # Safety
    ///
    /// self.deque must be valid.
    unsafe fn move_tail(&mut self, additional: usize) {
        let deque = unsafe { self.deque.as_mut() };
        let tail_start = deque.len + self.drain_len;
        deque.buf.reserve(tail_start + self.tail_len, additional);

        let new_tail_start = tail_start + additional;
        unsafe {
            deque.wrap_copy(tail_start, new_tail_start, self.tail_len);
        }
        self.drain_len += additional;
    }
}

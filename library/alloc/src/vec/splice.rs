use crate::alloc::{Allocator, Global};
use core::ptr::{self};
use core::slice::{self};

use super::{Drain, Vec};

/// A splicing iterator for `Vec`.
///
/// This struct is created by [`Vec::splice()`].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// let mut v = vec![0, 1, 2];
/// let new = [7, 8];
/// let iter: std::vec::Splice<'_, _> = v.splice(1.., new);
/// ```
#[derive(Debug)]
#[stable(feature = "vec_splice", since = "1.21.0")]
pub struct Splice<
    'a,
    I: Iterator + 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + 'a = Global,
> {
    pub(super) drain: Drain<'a, I::Item, A>,
    pub(super) replace_with: I,
}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator, A: Allocator> Iterator for Splice<'_, I, A> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator, A: Allocator> DoubleEndedIterator for Splice<'_, I, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator, A: Allocator> ExactSizeIterator for Splice<'_, I, A> {}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator, A: Allocator> Drop for Splice<'_, I, A> {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
        // At this point draining is done and the only remaining tasks are splicing
        // and moving things into the final place.
        // Which means we can replace the slice::Iter with pointers that won't point to deallocated
        // memory, so that Drain::drop is still allowed to call iter.len(), otherwise it would break
        // the ptr.sub_ptr contract.
        self.drain.iter = (&[]).iter();

        unsafe {
            if self.drain.tail_len == 0 {
                self.drain.vec.as_mut().extend(self.replace_with.by_ref());
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
        // Let `Drain::drop` move the tail back if necessary and restore `vec.len`.
    }
}

/// Private helper methods for `Splice::drop`
impl<T, A: Allocator> Drain<'_, T, A> {
    /// The range from `self.vec.len` to `self.tail_start` contains elements
    /// that have been moved out.
    /// Fill that range as much as possible with new elements from the `replace_with` iterator.
    /// Returns `true` if we filled the entire range. (`replace_with.next()` didn’t return `None`.)
    unsafe fn fill<I: Iterator<Item = T>>(&mut self, replace_with: &mut I) -> bool {
        let vec = unsafe { self.vec.as_mut() };
        let range_start = vec.len;
        let range_end = self.tail_start;
        let range_slice = unsafe {
            slice::from_raw_parts_mut(vec.as_mut_ptr().add(range_start), range_end - range_start)
        };

        for place in range_slice {
            if let Some(new_item) = replace_with.next() {
                unsafe { ptr::write(place, new_item) };
                vec.len += 1;
            } else {
                return false;
            }
        }
        true
    }

    /// Makes room for inserting more elements before the tail.
    unsafe fn move_tail(&mut self, additional: usize) {
        let vec = unsafe { self.vec.as_mut() };
        let len = self.tail_start + self.tail_len;
        vec.buf.reserve(len, additional);

        let new_tail_start = self.tail_start + additional;
        unsafe {
            let src = vec.as_ptr().add(self.tail_start);
            let dst = vec.as_mut_ptr().add(new_tail_start);
            ptr::copy(src, dst, self.tail_len);
        }
        self.tail_start = new_tail_start;
    }
}

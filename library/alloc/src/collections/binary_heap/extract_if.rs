use core::iter::FusedIterator;
use core::{fmt, ptr};

use super::BinaryHeap;
use crate::alloc::{Allocator, Global};

/// An iterator which uses a closure to determine if an element should be removed.
///
/// This struct is created by [`BinaryHeap::extract_if`].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// #![feature(binary_heap_extract_if)]
/// use crate::alloc::collections::BinaryHeap;
///
/// let mut heap: BinaryHeap<u32> = (0..128).collect();
/// let iter: Vec<u32> = heap.extract_if(|x| *x % 2 == 0).collect();
#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
#[must_use = "iterators are lazy and do nothing unless consumed; \
    use `retain_mut` or `extract_if().for_each(drop)` to remove and discard elements"]
pub struct ExtractIf<
    'a,
    T: Ord,
    F,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    heap: &'a mut BinaryHeap<T, A>,
    old_len: usize,
    del: usize,
    index: usize,
    predicate: F,
}

impl<T: Ord, F, A: Allocator> ExtractIf<'_, T, F, A> {
    pub(super) fn new<'a>(heap: &'a mut BinaryHeap<T, A>, predicate: F) -> ExtractIf<'a, T, F, A> {
        // This breaks the heap invariant but we artificially change the length to 0 below and don't change it back until we have fixed this invariant
        heap.sort_inner_vec();

        let old_len = heap.len();
        // SAFETY: leak enlargement
        unsafe { heap.data.set_len(0) };

        ExtractIf { heap, predicate, index: 0, old_len, del: 0 }
    }
}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A: Allocator> Iterator for ExtractIf<'_, T, F, A>
where
    F: FnMut(&T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.old_len {
            let i = self.index;
            // SAFETY:
            //  We know that `i < self.end` from the if guard and that `self.end <= self.old_len` from
            //  the validity of `Self`. Therefore `i` points to an element within `vec`.
            //
            //  Additionally, the i-th element is valid because each element is visited at most once
            //  and it is the first time we access vec[i].
            //
            //  Note: we can't use `vec.get_unchecked_mut(i)` here since the precondition for that
            //  function is that i < vec.len(), but we've set vec's length to zero.
            let cur = unsafe { &mut *self.heap.data.as_mut_ptr().add(i) };
            let extract = (self.predicate)(cur);
            // Update the index *after* the predicate is called. If the index
            // is updated prior and the predicate panics, the element at this
            // index would be leaked.
            self.index += 1;
            if extract {
                self.del += 1;
                // SAFETY: We never touch this element again after returning it.
                return Some(unsafe { ptr::read(cur) });
            } else if self.del > 0 {
                // SAFETY: `self.del` > 0, so the hole slot must not overlap with current element.
                // We use copy for move, and never touch this element again.
                unsafe {
                    let hole_slot = self.heap.data.as_mut_ptr().add(i - self.del);
                    ptr::copy_nonoverlapping(cur, hole_slot, 1);
                }
            }
        }
        None
    }
}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A: Allocator> Drop for ExtractIf<'_, T, F, A> {
    fn drop(&mut self) {
        if self.del > 0 {
            // SAFETY: Trailing unchecked items must be valid since we never touch them.
            unsafe {
                ptr::copy(
                    self.heap.data.as_ptr().add(self.index),
                    self.heap.data.as_mut_ptr().add(self.index - self.del),
                    self.old_len - self.index,
                );
            }
        }
        // SAFETY: After filling holes, all items are in contiguous memory.
        unsafe {
            self.heap.data.set_len(self.old_len - self.del);
        }
        self.heap.rebuild();
    }
}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A: Allocator> FusedIterator for ExtractIf<'_, T, F, A> where F: FnMut(&T) -> bool {}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A> fmt::Debug for ExtractIf<'_, T, F, A>
where
    T: fmt::Debug,
    A: Allocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf").finish_non_exhaustive()
    }
}

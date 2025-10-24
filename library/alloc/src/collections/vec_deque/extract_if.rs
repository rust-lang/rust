use core::ops::{Range, RangeBounds};
use core::{fmt, ptr, slice};

use super::VecDeque;
use crate::alloc::{Allocator, Global};

/// An iterator which uses a closure to determine if an element should be removed.
///
/// This struct is created by [`VecDeque::extract_if`].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// #![feature(vec_deque_extract_if)]
///
/// use std::collections::vec_deque::ExtractIf;
/// use std::collections::vec_deque::VecDeque;
///
/// let mut v = VecDeque::from([0, 1, 2]);
/// let iter: ExtractIf<'_, _, _> = v.extract_if(.., |x| *x % 2 == 0);
/// ```
#[unstable(feature = "vec_deque_extract_if", issue = "147750")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<
    'a,
    T,
    F,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    vec: &'a mut VecDeque<T, A>,
    /// The index of the item that will be inspected by the next call to `next`.
    idx: usize,
    /// Elements at and beyond this point will be retained. Must be equal or smaller than `old_len`.
    end: usize,
    /// The number of items that have been drained (removed) thus far.
    del: usize,
    /// The original length of `vec` prior to draining.
    old_len: usize,
    /// The filter test predicate.
    pred: F,
}

impl<'a, T, F, A: Allocator> ExtractIf<'a, T, F, A> {
    pub(super) fn new<R: RangeBounds<usize>>(
        vec: &'a mut VecDeque<T, A>,
        pred: F,
        range: R,
    ) -> Self {
        let old_len = vec.len();
        let Range { start, end } = slice::range(range, ..old_len);

        // Guard against the deque getting leaked (leak amplification)
        vec.len = 0;
        ExtractIf { vec, idx: start, del: 0, end, old_len, pred }
    }

    /// Returns a reference to the underlying allocator.
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn allocator(&self) -> &A {
        self.vec.allocator()
    }
}

#[unstable(feature = "vec_deque_extract_if", issue = "147750")]
impl<T, F, A: Allocator> Iterator for ExtractIf<'_, T, F, A>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while self.idx < self.end {
            let i = self.idx;
            // SAFETY:
            //  We know that `i < self.end` from the if guard and that `self.end <= self.old_len` from
            //  the validity of `Self`. Therefore `i` points to an element within `vec`.
            //
            //  Additionally, the i-th element is valid because each element is visited at most once
            //  and it is the first time we access vec[i].
            //
            //  Note: we can't use `vec.get_mut(i).unwrap()` here since the precondition for that
            //  function is that i < vec.len, but we've set vec's length to zero.
            let idx = self.vec.to_physical_idx(i);
            let cur = unsafe { &mut *self.vec.ptr().add(idx) };
            let drained = (self.pred)(cur);
            // Update the index *after* the predicate is called. If the index
            // is updated prior and the predicate panics, the element at this
            // index would be leaked.
            self.idx += 1;
            if drained {
                self.del += 1;
                // SAFETY: We never touch this element again after returning it.
                return Some(unsafe { ptr::read(cur) });
            } else if self.del > 0 {
                let hole_slot = self.vec.to_physical_idx(i - self.del);
                // SAFETY: `self.del` > 0, so the hole slot must not overlap with current element.
                // We use copy for move, and never touch this element again.
                unsafe { self.vec.wrap_copy(idx, hole_slot, 1) };
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.end - self.idx))
    }
}

#[unstable(feature = "vec_deque_extract_if", issue = "147750")]
impl<T, F, A: Allocator> Drop for ExtractIf<'_, T, F, A> {
    fn drop(&mut self) {
        if self.del > 0 {
            let src = self.vec.to_physical_idx(self.idx);
            let dst = self.vec.to_physical_idx(self.idx - self.del);
            let len = self.old_len - self.idx;
            // SAFETY: Trailing unchecked items must be valid since we never touch them.
            unsafe { self.vec.wrap_copy(src, dst, len) };
        }
        self.vec.len = self.old_len - self.del;
    }
}

#[unstable(feature = "vec_deque_extract_if", issue = "147750")]
impl<T, F, A> fmt::Debug for ExtractIf<'_, T, F, A>
where
    T: fmt::Debug,
    A: Allocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let peek = if self.idx < self.end {
            let idx = self.vec.to_physical_idx(self.idx);
            // This has to use pointer arithmetic as `self.vec[self.idx]` or
            // `self.vec.get_unchecked(self.idx)` wouldn't work since we
            // temporarily set the length of `self.vec` to zero.
            //
            // SAFETY:
            // Since `self.idx` is smaller than `self.end` and `self.end` is
            // smaller than `self.old_len`, `idx` is valid for indexing the
            // buffer. Also, per the invariant of `self.idx`, this element
            // has not been inspected/moved out yet.
            Some(unsafe { &*self.vec.ptr().add(idx) })
        } else {
            None
        };
        f.debug_struct("ExtractIf").field("peek", &peek).finish_non_exhaustive()
    }
}

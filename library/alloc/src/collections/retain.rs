use core::{cmp, hint, mem, ptr};

/// # Safety
///
/// There are `front_len` slots in `front` and if `index >= front_len`,
/// there are `index - front_len + 1` slots in `back` at least.
unsafe fn slot<T>(front: *mut T, front_len: usize, back: *mut T, index: usize) -> *mut T {
    // SAFETY: Upheld by caller.
    unsafe { if index < front_len { front.add(index) } else { back.add(index - front_len) } }
}

/// Retains elements in two contiguous regions according to `f`.
///
/// The `*total_len` initialized elements are treated as one logical sequence:
/// the first `front_len` elements start at `front`, and the remaining elements
/// start at `back`.
///
/// Each element is passed to `f` in logical order. Elements for which `f`
/// returns `true` are compacted toward the beginning of the logical sequence,
/// preserving their relative order. Rejected elements are dropped.
///
/// On return, `*total_len` is updated to the number of retained elements.
///
/// On panic, retained and unchecked elements, including the element whose
/// predicate panicked, are compacted and `*total_len` is updated accordingly.
///
/// # Safety
///
/// - `front` must point to `front_len` initialized, valid instances of `T`.
/// - `back` must point to `*total_len - front_len` initialized, valid instances
///   of `T`.
/// - `front_len <= *total_len`.
/// - The two regions must not overlap.
/// - All elements in both regions must be exclusively accessible for the
///   duration of this call.
/// - `total_len` must not point into either region.
#[inline(always)]
pub(crate) unsafe fn retain_raw<T, F>(
    front: *mut T,
    front_len: usize,
    back: *mut T,
    total_len: &mut usize,
    mut f: F,
) where
    F: FnMut(&mut T) -> bool,
{
    let original_len = *total_len;
    if original_len == 0 {
        return;
    }

    // Physical layout:
    //
    // front: [ ......... front_len elements ......... ]
    // back:  [ .... original_len - front_len elements .... ]
    //
    // Logical layout:
    //
    // [Kept, Kept, Hole, Hole, Hole, Unchecked, Unchecked]
    //              ^write            ^read
    // |<--------------- original_len ------------------->|
    //
    // `front` contains logical indices `[0, front_len)`, and `back` contains
    // `[front_len, original_len)`. `read` and `write` are logical offsets and may
    // therefore refer to either physical region.
    //
    // Kept: Elements for which the predicate returned true.
    // Hole: Slots whose elements have been moved from or dropped.
    // Unchecked: Initialized elements that have not been visited yet.
    //
    // This drop guard will be invoked when predicate or `drop` of element panicked.
    // It shifts unchecked elements to cover holes and assigns the correct length to `*total_len`.
    // In cases when predicate and `drop` never panic, it will be optimized out.
    struct PanicGuard<'a, T> {
        front: *mut T,
        front_len: usize,
        back: *mut T,
        total_len: &'a mut usize,
        read: usize,
        write: usize,
        original_len: usize,
    }

    impl<T> Drop for PanicGuard<'_, T> {
        #[cold]
        fn drop(&mut self) {
            let remaining = self.original_len - self.read;
            for i in 0..remaining {
                // SAFETY: `write < read`, so the slots don't overlap.
                // Trailing unchecked items must be valid since we never touch them.
                unsafe {
                    ptr::copy_nonoverlapping(
                        slot(self.front, self.front_len, self.back, self.read + i),
                        slot(self.front, self.front_len, self.back, self.write + i),
                        1,
                    )
                };
            }
            *self.total_len = self.write + remaining;
        }
    }

    /// # Safety
    ///
    /// The `base` must have `n` initialized elements at least and
    /// the region must be exclusively accessible for the duration of this call.
    #[inline(always)]
    unsafe fn first_rejected<T, F>(base: *mut T, n: usize, f: &mut F) -> Option<usize>
    where
        F: FnMut(&mut T) -> bool,
    {
        for i in 0..n {
            // SAFETY: `i < n`
            if hint::unlikely(!f(unsafe { &mut *base.add(i) })) {
                return Some(i);
            }
        }
        None
    }

    /// It calls `f` at `src` and moves retained element to `dst`
    /// until `g.read` reaches `end`.
    ///
    /// # Safety
    /// - `src` and `dst` must be the physical addresses of `g.read` and `g.write`.
    /// - `src` contains `end - g.read` elements at least and
    ///   `dst` contains `end - g.read` slots at least.
    /// - `end <= g.original_len`
    #[inline(always)]
    unsafe fn compact<T, F>(
        g: &mut PanicGuard<'_, T>,
        end: usize,
        src: *mut T,
        dst: *mut T,
        f: &mut F,
    ) where
        F: FnMut(&mut T) -> bool,
    {
        let src_start = g.read;
        let dst_start = g.write;
        while g.read < end {
            // SAFETY: `g.read < end` and `src` has `end - src_start` elements at least.
            let cur = unsafe { &mut *src.add(g.read - src_start) };
            if !f(cur) {
                // Advance `read` early to avoid double drop if `drop_in_place` panicked.
                g.read += 1;
                // SAFETY: We never touch this element again after dropped.
                unsafe { ptr::drop_in_place(cur) };
            } else {
                // SAFETY: `g.write < g.read < end`, so the slots don't overlap.
                // We use copy for move, and never touch the source element again.
                unsafe { ptr::copy_nonoverlapping(cur, dst.add(g.write - dst_start), 1) };

                g.write += 1;
                g.read += 1;
            }
        }
    }

    let back_len = original_len - front_len;

    // SAFETY: guaranteed by caller
    let (read, rejected) = unsafe {
        if let Some(n) = first_rejected(front, front_len, &mut f) {
            (n, front.add(n))
        } else if let Some(n) = first_rejected(back, back_len, &mut f) {
            (front_len + n, back.add(n))
        } else {
            // All elements are kept, return early.
            return;
        }
    };

    // Critical section starts here and at least one element is going to be removed.
    // Advance `g.read` early to avoid double drop if `drop_in_place` panicked.
    let mut g =
        PanicGuard { front, back, front_len, total_len, read: read + 1, write: read, original_len };
    // SAFETY: `rejected` belongs to either `front` or `back` and is initialized.
    unsafe { ptr::drop_in_place(rejected) };

    // Retains the front.
    if g.read < front_len {
        let (read, write) = (g.read, g.write);
        // SAFETY: `front` has `front_len` elements; `write < read` and both is in the range of `front`.
        unsafe { compact(&mut g, front_len, front.add(read), front.add(write), &mut f) };
    }

    // Moves elements of back to front if there still are holes in front.
    while g.write < front_len && g.read < original_len {
        let end = g.read + cmp::min(original_len - g.read, front_len - g.write);
        let (read, write) = (g.read, g.write);
        // SAFETY: It accesses elements up to `end - g.read`,
        // which is the minimum of both accessible `front` and `back`.
        // The physical `read` has been at `back` now, so `read - front_len` isn't negative.
        unsafe { compact(&mut g, end, back.add(read - front_len), front.add(write), &mut f) };
    }

    // Retains the back.
    if g.read < original_len {
        let (read, write) = (g.read, g.write);
        // SAFETY: `back` remains `original_len - g.read` unread elements.
        unsafe {
            compact(
                &mut g,
                original_len,
                back.add(read - front_len),
                back.add(write - front_len),
                &mut f,
            )
        };
    }

    // We are leaving the critical section and no panic happened.
    // Commit the length change and forget the guard.
    *g.total_len = g.write;
    mem::forget(g);
}

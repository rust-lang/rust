use core::{cmp, hint, mem, ptr};

/// Returns the physical index corresponding to a logical `index`.
///
/// Requires `index <= cap` and `head < cap`.
///
/// For ZSTs, `head + index` may wrap around `usize::MAX`. In that case, the
/// returned value may be one less than the mathematically correct wrapped
/// index. This is harmless because all ZST elements have the same address, and
/// `cap - phys(...)` remains non-zero, which is sufficient for the loop in
/// `retain_raw`.
#[inline(always)]
const fn phys(head: usize, cap: usize, index: usize) -> usize {
    let index = head.wrapping_add(index);
    if index >= cap { index - cap } else { index }
}

/// # Safety
///
/// - `buf` must point to an allocation containing at least `cap` slots for `T`.
/// - `index <= cap` and `head < cap`.
#[inline(always)]
const unsafe fn slot<T>(buf: *mut T, head: usize, cap: usize, index: usize) -> *mut T {
    // SAFETY: Upheld by caller.
    unsafe { buf.add(phys(head, cap, index)) }
}

/// Retains elements in a ring buffer according to `f`.
///
/// The `len` initialized elements starting at `head` are treated as one
/// logically contiguous sequence. Logical indices that reach `cap` wrap
/// around to the beginning of `buf`.
///
/// Each element is passed to `f` in logical order. Elements for which `f`
/// returns `true` are compacted toward `head`, preserving their relative
/// order. Rejected elements are dropped.
///
/// On return, `*len` is updated to the number of retained elements. The first
/// `*len` logical slots starting at `head` contain the retained elements.
///
/// On panic, this compacts retained and unchecked elements, including the
/// element whose predicate panicked, updates `*len`, and leaves `head` unchanged.
///
/// # Safety
///
/// - `buf` must point to an allocation containing at least `cap` slots for `T`.
/// - The `*len` logical slots starting at `head` must contain initialized,
///   valid instances of `T`.
/// - `*len <= cap`.
/// - If `*len != 0`, then `head < cap`.
/// - The elements in the logical range `[0, *len)` must be exclusively
///   accessible for the duration of this call.
/// - `len` must not point into `buf`.
pub(crate) unsafe fn retain_raw<T, F>(
    buf: *mut T,
    head: usize,
    cap: usize,
    len: &mut usize,
    mut f: F,
) where
    F: FnMut(&mut T) -> bool,
{
    let original_len = *len;
    if original_len == 0 {
        return;
    }

    // Physical layout:
    //
    // buf[0]                                                                        buf[cap - 1]
    //   |                                                                                  |
    //   v                                                                                  v
    // [Unchecked, Unchecked, Uninitialized, ...., Uninitialized, Kept, Kept, Hole, Hole, Hole]
    //  ^                                                         ^head
    //  |                   |
    //  +--- wrapped part --+
    //
    // Logical order starting at `head`:
    //
    // [Kept, Kept, Hole, Hole, Hole, Unchecked, Unchecked]
    //              ^write            ^read
    // |<--------------- original_len ------------------->|
    //
    // `head` is a physical index. `read` and `write` are logical offsets from
    // `head`; they are mapped into `buf` with `phys(head, cap, ...)`.
    //
    // Kept: Elements which predicate returns true on.
    // Hole: Moved or dropped element slot.
    // Unchecked: Unchecked valid elements.
    // Uninitialized: Uninitialized element slot, which must not be used.
    //
    // This drop guard will be invoked when predicate or `drop` of element panicked.
    // It shifts unchecked elements to cover holes and assigns the correct length to `*len`.
    // In cases when predicate and `drop` never panic, it will be optimized out.
    struct PanicGuard<'a, T> {
        buf: *mut T,
        head: usize,
        cap: usize,
        len: &'a mut usize,
        read: usize,
        write: usize,
        original_len: usize,
    }

    impl<T> Drop for PanicGuard<'_, T> {
        #[cold]
        fn drop(&mut self) {
            let remaining = self.original_len - self.read;
            let (buf, head, cap) = (self.buf, self.head, self.cap);
            for i in 0..remaining {
                // SAFETY: `0 < read - write < cap`, so the slots don't overlap.
                // Trailing unchecked items must be valid since we never touch them.
                unsafe {
                    ptr::copy_nonoverlapping(
                        slot(buf, head, cap, self.read + i),
                        slot(buf, head, cap, self.write + i),
                        1,
                    )
                };
            }
            *self.len = self.write + remaining;
        }
    }

    let mut read = 0;
    loop {
        // Compute the longest contiguous run before the read position wraps around.
        // `slot` handles the wrap on the next iteration.
        let n = cmp::min(original_len - read, cap - phys(head, cap, read));
        // SAFETY: read < original_len
        let base = unsafe { slot(buf, head, cap, read) };

        if let Some(i) = (0..n).position(|i| {
            // SAFETY: `i < n`, so this stays within the accessible contiguous range
            // `[phys(read), min(phys(read) + (original_len - read), cap))`.
            hint::unlikely(!f(unsafe { &mut *base.add(i) }))
        }) {
            read += i;
            break;
        }
        read += n;
        if read == original_len {
            // All elements are kept, return early.
            return;
        }
    }

    // Critical section starts here and at least one element is going to be removed.
    // Advance `g.read` early to avoid double drop if `drop_in_place` panicked.
    let mut g = PanicGuard { buf, head, cap, len, read: read + 1, write: read, original_len };
    // SAFETY: previous `read` is always less than original_len.
    unsafe { ptr::drop_in_place(slot(buf, head, cap, read)) };

    while g.read < g.original_len {
        // Compute the longest contiguous run before either the read or write position
        // wraps around the ring buffer. The remaining logical length bounds the run to
        // initialized elements.
        let run = cmp::min(
            g.original_len - g.read,
            cmp::min(cap - phys(head, cap, g.read), cap - phys(head, cap, g.write)),
        );
        // SAFETY: `read` is always less than original_len.
        let src = unsafe { slot(buf, head, cap, g.read) };
        // SAFETY: `write < read < original_len <= cap`.
        let mut hole = unsafe { slot(buf, head, cap, g.write) };
        for i in 0..run {
            // SAFETY: `i < run`, so this stays within the accessible contiguous range
            // `[phys(read), min(phys(read) + (original_len - read), cap))`.
            let cur = unsafe { &mut *src.add(i) };
            if !f(cur) {
                // Advance `read` early to avoid double drop if `drop_in_place` panicked.
                g.read += 1;
                // SAFETY: We never touch this element again after dropped.
                unsafe { ptr::drop_in_place(cur) };
            } else {
                // SAFETY: `0 < read - write < cap`, so the slots don't overlap.
                // We use copy for move, and never touch the source element again.
                unsafe { ptr::copy_nonoverlapping(cur, hole, 1) };
                // SAFETY: The maximum is `buf + cap`.
                hole = unsafe { hole.add(1) };
                g.write += 1;
                g.read += 1;
            }
        }
    }

    // We are leaving the critical section and no panic happened.
    // Commit the length change and forget the guard.
    *g.len = g.write;
    mem::forget(g);
}

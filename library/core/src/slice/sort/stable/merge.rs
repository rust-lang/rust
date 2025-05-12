//! This module contains logic for performing a merge of two sorted sub-slices.

use crate::mem::MaybeUninit;
use crate::{cmp, ptr};

/// Merges non-decreasing runs `v[..mid]` and `v[mid..]` using `scratch` as
/// temporary storage, and stores the result into `v[..]`.
pub fn merge<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mid: usize,
    is_less: &mut F,
) {
    let len = v.len();

    if mid == 0 || mid >= len || scratch.len() < cmp::min(mid, len - mid) {
        return;
    }

    // SAFETY: We checked that the two slices are non-empty and `mid` is in-bounds.
    // We checked that the buffer `scratch` has enough capacity to hold a copy of
    // the shorter slice. `merge_up` and `merge_down` are written in such a way that
    // they uphold the contract described in `MergeState::drop`.
    unsafe {
        // The merge process first copies the shorter run into `buf`. Then it traces
        // the newly copied run and the longer run forwards (or backwards), comparing
        // their next unconsumed elements and copying the lesser (or greater) one into `v`.
        //
        // As soon as the shorter run is fully consumed, the process is done. If the
        // longer run gets consumed first, then we must copy whatever is left of the
        // shorter run into the remaining gap in `v`.
        //
        // Intermediate state of the process is always tracked by `gap`, which serves
        // two purposes:
        //  1. Protects integrity of `v` from panics in `is_less`.
        //  2. Fills the remaining gap in `v` if the longer run gets consumed first.

        let buf = MaybeUninit::slice_as_mut_ptr(scratch);

        let v_base = v.as_mut_ptr();
        let v_mid = v_base.add(mid);
        let v_end = v_base.add(len);

        let left_len = mid;
        let right_len = len - mid;

        let left_is_shorter = left_len <= right_len;
        let save_base = if left_is_shorter { v_base } else { v_mid };
        let save_len = if left_is_shorter { left_len } else { right_len };

        ptr::copy_nonoverlapping(save_base, buf, save_len);

        let mut merge_state = MergeState { start: buf, end: buf.add(save_len), dst: save_base };

        if left_is_shorter {
            merge_state.merge_up(v_mid, v_end, is_less);
        } else {
            merge_state.merge_down(v_base, buf, v_end, is_less);
        }
        // Finally, `merge_state` gets dropped. If the shorter run was not fully
        // consumed, whatever remains of it will now be copied into the hole in `v`.
    }
}

// When dropped, copies the range `start..end` into `dst..`.
struct MergeState<T> {
    start: *mut T,
    end: *mut T,
    dst: *mut T,
}

impl<T> MergeState<T> {
    /// # Safety
    /// The caller MUST guarantee that `self` is initialized in a way where `start -> end` is
    /// the longer sub-slice and so that `dst` can be written to at least the shorter sub-slice
    /// length times. In addition `start -> end` and `right -> right_end` MUST be valid to be
    /// read. This function MUST only be called once.
    unsafe fn merge_up<F: FnMut(&T, &T) -> bool>(
        &mut self,
        mut right: *const T,
        right_end: *const T,
        is_less: &mut F,
    ) {
        // SAFETY: See function safety comment.
        unsafe {
            let left = &mut self.start;
            let out = &mut self.dst;

            while *left != self.end && right as *const T != right_end {
                let consume_left = !is_less(&*right, &**left);

                let src = if consume_left { *left } else { right };
                ptr::copy_nonoverlapping(src, *out, 1);

                *left = left.add(consume_left as usize);
                right = right.add(!consume_left as usize);

                *out = out.add(1);
            }
        }
    }

    /// # Safety
    /// The caller MUST guarantee that `self` is initialized in a way where `left_end <- dst` is
    /// the shorter sub-slice and so that `out` can be written to at least the shorter sub-slice
    /// length times. In addition `left_end <- dst` and `right_end <- end` MUST be valid to be
    /// read. This function MUST only be called once.
    unsafe fn merge_down<F: FnMut(&T, &T) -> bool>(
        &mut self,
        left_end: *const T,
        right_end: *const T,
        mut out: *mut T,
        is_less: &mut F,
    ) {
        // SAFETY: See function safety comment.
        unsafe {
            loop {
                let left = self.dst.sub(1);
                let right = self.end.sub(1);
                out = out.sub(1);

                let consume_left = is_less(&*right, &*left);

                let src = if consume_left { left } else { right };
                ptr::copy_nonoverlapping(src, out, 1);

                self.dst = left.add(!consume_left as usize);
                self.end = right.add(consume_left as usize);

                if self.dst as *const T == left_end || self.end as *const T == right_end {
                    break;
                }
            }
        }
    }
}

impl<T> Drop for MergeState<T> {
    fn drop(&mut self) {
        // SAFETY: The user of MergeState MUST ensure, that at any point this drop
        // impl MAY run, for example when the user provided `is_less` panics, that
        // copying the contiguous region between `start` and `end` to `dst` will
        // leave the input slice `v` with each original element and all possible
        // modifications observed.
        unsafe {
            let len = self.end.offset_from_unsigned(self.start);
            ptr::copy_nonoverlapping(self.start, self.dst, len);
        }
    }
}

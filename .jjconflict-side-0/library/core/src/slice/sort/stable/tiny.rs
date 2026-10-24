//! Binary-size optimized mergesort inspired by https://github.com/voultapher/tiny-sort-rs.

use crate::mem::MaybeUninit;
use crate::ptr;
use crate::slice::sort::stable::merge;

/// Tiny recursive top-down merge sort optimized for binary size. It has no adaptiveness whatsoever,
/// no run detection, etc.
#[inline(always)]
pub fn mergesort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    is_less: &mut F,
) {
    let len = v.len();

    if len > 2 {
        let mid = len / 2;

        // SAFETY: mid is in-bounds.
        unsafe {
            // Sort the left half recursively.
            mergesort(v.get_unchecked_mut(..mid), scratch, is_less);
            // Sort the right half recursively.
            mergesort(v.get_unchecked_mut(mid..), scratch, is_less);
        }

        merge::merge(v, scratch, mid, is_less);
    } else if len == 2 {
        // SAFETY: We checked the len, the pointers we create are valid and don't overlap.
        unsafe {
            let v_base = v.as_mut_ptr();
            let v_a = v_base;
            let v_b = v_base.add(1);

            if is_less(&*v_b, &*v_a) {
                ptr::swap_nonoverlapping(v_a, v_b, 1);
            }
        }
    }
}

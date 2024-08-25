//! Binary-size optimized mergesort inspired by https://github.com/voultapher/tiny-sort-rs.

use crate::mem::{ManuallyDrop, MaybeUninit};
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
        // Branchless swap the two elements. This reduces the recursion depth and improves
        // perf significantly at a small binary-size cost. Trades ~10% perf boost for integers
        // for ~50 bytes in the binary.

        // SAFETY: We checked the len, the pointers we create are valid and don't overlap.
        unsafe {
            swap_if_less(v.as_mut_ptr(), 0, 1, is_less);
        }
    }
}

/// Swap two values in the slice pointed to by `v_base` at the position `a_pos` and `b_pos` if the
/// value at position `b_pos` is less than the one at position `a_pos`.
unsafe fn swap_if_less<T, F>(v_base: *mut T, a_pos: usize, b_pos: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: the caller must guarantee that `a` and `b` each added to `v_base` yield valid
    // pointers into `v_base`, and are properly aligned, and part of the same allocation.
    unsafe {
        let v_a = v_base.add(a_pos);
        let v_b = v_base.add(b_pos);

        // PANIC SAFETY: if is_less panics, no scratch memory was created and the slice should still be
        // in a well defined state, without duplicates.

        // Important to only swap if it is more and not if it is equal. is_less should return false for
        // equal, so we don't swap.
        let should_swap = is_less(&*v_b, &*v_a);

        // This is a branchless version of swap if.
        // The equivalent code with a branch would be:
        //
        // if should_swap {
        //     ptr::swap(left, right, 1);
        // }

        // The goal is to generate cmov instructions here.
        let left_swap = if should_swap { v_b } else { v_a };
        let right_swap = if should_swap { v_a } else { v_b };

        let right_swap_tmp = ManuallyDrop::new(ptr::read(right_swap));
        ptr::copy(left_swap, v_a, 1);
        ptr::copy_nonoverlapping(&*right_swap_tmp, v_b, 1);
    }
}

//! This module contains a branchless heapsort as fallback for unstable quicksort.

use crate::{cmp, intrinsics, ptr};

/// Sorts `v` using heapsort, which guarantees *O*(*n* \* log(*n*)) worst-case.
///
/// Never inline this, it sits the main hot-loop in `recurse` and is meant as unlikely algorithmic
/// fallback.
#[inline(never)]
pub(crate) fn heapsort<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    for i in (0..len + len / 2).rev() {
        let sift_idx = if i >= len {
            i - len
        } else {
            v.swap(0, i);
            0
        };

        // SAFETY: The above calculation ensures that `sift_idx` is either 0 or
        // `(len..(len + (len / 2))) - len`, which simplifies to `0..(len / 2)`.
        // This guarantees the required `sift_idx <= len`.
        unsafe {
            sift_down(&mut v[..cmp::min(i, len)], sift_idx, is_less);
        }
    }
}

// This binary heap respects the invariant `parent >= child`.
//
// SAFETY: The caller has to guarantee that `node <= v.len()`.
#[inline(always)]
unsafe fn sift_down<T, F>(v: &mut [T], mut node: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: See function safety.
    unsafe {
        intrinsics::assume(node <= v.len());
    }

    let len = v.len();

    let v_base = v.as_mut_ptr();

    loop {
        // Children of `node`.
        let mut child = 2 * node + 1;
        if child >= len {
            break;
        }

        // SAFETY: The invariants and checks guarantee that both node and child are in-bounds.
        unsafe {
            // Choose the greater child.
            if child + 1 < len {
                // We need a branch to be sure not to out-of-bounds index,
                // but it's highly predictable.  The comparison, however,
                // is better done branchless, especially for primitives.
                child += is_less(&*v_base.add(child), &*v_base.add(child + 1)) as usize;
            }

            // Stop if the invariant holds at `node`.
            if !is_less(&*v_base.add(node), &*v_base.add(child)) {
                break;
            }

            ptr::swap_nonoverlapping(v_base.add(node), v_base.add(child), 1);
        }

        node = child;
    }
}

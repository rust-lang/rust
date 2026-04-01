#![cfg_attr(any(feature = "optimize_for_size", target_pointer_width = "16"), allow(dead_code))]

use crate::marker::Freeze;

pub(crate) mod pivot;
pub(crate) mod smallsort;

/// SAFETY: this is safety relevant, how does this interact with the soundness holes in
/// specialization?
#[rustc_unsafe_specialization_marker]
pub(crate) trait FreezeMarker {}

impl<T: Freeze> FreezeMarker for T {}

/// Finds a run of sorted elements starting at the beginning of the slice.
///
/// Returns the length of the run, and a bool that is false when the run
/// is ascending, and true if the run strictly descending.
#[inline(always)]
pub(crate) fn find_existing_run<T, F: FnMut(&T, &T) -> bool>(
    v: &[T],
    is_less: &mut F,
) -> (usize, bool) {
    let len = v.len();
    if len < 2 {
        return (len, false);
    }

    // SAFETY: We checked that len >= 2, so 0 and 1 are valid indices.
    // This also means that run_len < len implies run_len and run_len - 1
    // are valid indices as well.
    unsafe {
        let mut run_len = 2;
        let strictly_descending = is_less(v.get_unchecked(1), v.get_unchecked(0));
        if strictly_descending {
            while run_len < len && is_less(v.get_unchecked(run_len), v.get_unchecked(run_len - 1)) {
                run_len += 1;
            }
        } else {
            while run_len < len && !is_less(v.get_unchecked(run_len), v.get_unchecked(run_len - 1))
            {
                run_len += 1;
            }
        }
        (run_len, strictly_descending)
    }
}

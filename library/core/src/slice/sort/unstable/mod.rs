//! This module contains the entry points for `slice::sort_unstable`.

use crate::intrinsics;
use crate::mem::SizedTypeProperties;
#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
use crate::slice::sort::shared::find_existing_run;
#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
use crate::slice::sort::shared::smallsort::insertion_sort_shift_left;

pub(crate) mod heapsort;
pub(crate) mod quicksort;

/// Unstable sort called ipnsort by Lukas Bergdoll and Orson Peters.
/// Design document:
/// <https://github.com/Voultapher/sort-research-rs/blob/main/writeup/ipnsort_introduction/text.md>
///
/// Upholds all safety properties outlined here:
/// <https://github.com/Voultapher/sort-research-rs/blob/main/writeup/sort_safety/text.md>
#[inline(always)]
pub fn sort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], is_less: &mut F) {
    // Arrays of zero-sized types are always all-equal, and thus sorted.
    if T::IS_ZST {
        return;
    }

    // Instrumenting the standard library showed that 90+% of the calls to sort
    // by rustc are either of size 0 or 1.
    let len = v.len();
    if intrinsics::likely(len < 2) {
        return;
    }

    cfg_if! {
        if #[cfg(any(feature = "optimize_for_size", target_pointer_width = "16"))] {
            heapsort::heapsort(v, is_less);
        } else {
            // More advanced sorting methods than insertion sort are faster if called in
            // a hot loop for small inputs, but for general-purpose code the small
            // binary size of insertion sort is more important. The instruction cache in
            // modern processors is very valuable, and for a single sort call in general
            // purpose code any gains from an advanced method are cancelled by i-cache
            // misses during the sort, and thrashing the i-cache for surrounding code.
            const MAX_LEN_ALWAYS_INSERTION_SORT: usize = 20;
            if intrinsics::likely(len <= MAX_LEN_ALWAYS_INSERTION_SORT) {
                insertion_sort_shift_left(v, 1, is_less);
                return;
            }

            ipnsort(v, is_less);
        }
    }
}

/// See [`sort`]
///
/// Deliberately don't inline the main sorting routine entrypoint to ensure the
/// inlined insertion sort i-cache footprint remains minimal.
#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
#[inline(never)]
fn ipnsort<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    let (run_len, was_reversed) = find_existing_run(v, is_less);

    // SAFETY: find_existing_run promises to return a valid run_len.
    unsafe { intrinsics::assume(run_len <= len) };

    if run_len == len {
        if was_reversed {
            v.reverse();
        }

        // It would be possible to a do in-place merging here for a long existing streak. But that
        // makes the implementation a lot bigger, users can use `slice::sort` for that use-case.
        return;
    }

    // Limit the number of imbalanced partitions to `2 * floor(log2(len))`.
    // The binary OR by one is used to eliminate the zero-check in the logarithm.
    let limit = 2 * (len | 1).ilog2();
    crate::slice::sort::unstable::quicksort::quicksort(v, None, limit, is_less);
}

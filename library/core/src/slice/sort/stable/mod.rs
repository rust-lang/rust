//! This module contains the entry points for `slice::sort`.

#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
use crate::cmp;
use crate::mem::{MaybeUninit, SizedTypeProperties};
#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
use crate::slice::sort::shared::smallsort::{
    SMALL_SORT_GENERAL_SCRATCH_LEN, StableSmallSortTypeImpl, insertion_sort_shift_left,
};
use crate::{cfg_match, intrinsics};

pub(crate) mod merge;

#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
pub(crate) mod drift;
#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
pub(crate) mod quicksort;

#[cfg(any(feature = "optimize_for_size", target_pointer_width = "16"))]
pub(crate) mod tiny;

/// Stable sort called driftsort by Orson Peters and Lukas Bergdoll.
/// Design document:
/// <https://github.com/Voultapher/sort-research-rs/blob/main/writeup/driftsort_introduction/text.md>
///
/// Upholds all safety properties outlined here:
/// <https://github.com/Voultapher/sort-research-rs/blob/main/writeup/sort_safety/text.md>
#[inline(always)]
pub fn sort<T, F: FnMut(&T, &T) -> bool, BufT: BufGuard<T>>(v: &mut [T], is_less: &mut F) {
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

    cfg_match! {
        any(feature = "optimize_for_size", target_pointer_width = "16") => {
            // Unlike driftsort, mergesort only requires len / 2,
            // not len - len / 2.
            let alloc_len = len / 2;

            cfg_match! {
                target_pointer_width = "16" => {
                    let mut heap_buf = BufT::with_capacity(alloc_len);
                    let scratch = heap_buf.as_uninit_slice_mut();
                }
                _ => {
                    // For small inputs 4KiB of stack storage suffices, which allows us to avoid
                    // calling the (de-)allocator. Benchmarks showed this was quite beneficial.
                    let mut stack_buf = AlignedStorage::<T, 4096>::new();
                    let stack_scratch = stack_buf.as_uninit_slice_mut();
                    let mut heap_buf;
                    let scratch = if stack_scratch.len() >= alloc_len {
                        stack_scratch
                    } else {
                        heap_buf = BufT::with_capacity(alloc_len);
                        heap_buf.as_uninit_slice_mut()
                    };
                }
            }

            tiny::mergesort(v, scratch, is_less);
        }
        _ => {
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

            driftsort_main::<T, F, BufT>(v, is_less);
        }
    }
}

/// See [`sort`]
///
/// Deliberately don't inline the main sorting routine entrypoint to ensure the
/// inlined insertion sort i-cache footprint remains minimal.
#[cfg(not(any(feature = "optimize_for_size", target_pointer_width = "16")))]
#[inline(never)]
fn driftsort_main<T, F: FnMut(&T, &T) -> bool, BufT: BufGuard<T>>(v: &mut [T], is_less: &mut F) {
    // By allocating n elements of memory we can ensure the entire input can
    // be sorted using stable quicksort, which allows better performance on
    // random and low-cardinality distributions. However, we still want to
    // reduce our memory usage to n - n / 2 for large inputs. We do this by scaling
    // our allocation as max(n - n / 2, min(n, 8MB)), ensuring we scale like n for
    // small inputs and n - n / 2 for large inputs, without a sudden drop off. We
    // also need to ensure our alloc >= SMALL_SORT_GENERAL_SCRATCH_LEN, as the
    // small-sort always needs this much memory.
    //
    // driftsort will produce unsorted runs of up to min_good_run_len, which
    // is at most len - len / 2.
    // Unsorted runs need to be processed by quicksort, which requires as much
    // scratch space as the run length, therefore the scratch space must be at
    // least len - len / 2.
    // If min_good_run_len is ever modified, this code must be updated to allocate
    // the correct scratch size for it.
    const MAX_FULL_ALLOC_BYTES: usize = 8_000_000; // 8MB
    let max_full_alloc = MAX_FULL_ALLOC_BYTES / size_of::<T>();
    let len = v.len();
    let alloc_len = cmp::max(
        cmp::max(len - len / 2, cmp::min(len, max_full_alloc)),
        SMALL_SORT_GENERAL_SCRATCH_LEN,
    );

    // For small inputs 4KiB of stack storage suffices, which allows us to avoid
    // calling the (de-)allocator. Benchmarks showed this was quite beneficial.
    let mut stack_buf = AlignedStorage::<T, 4096>::new();
    let stack_scratch = stack_buf.as_uninit_slice_mut();
    let mut heap_buf;
    let scratch = if stack_scratch.len() >= alloc_len {
        stack_scratch
    } else {
        heap_buf = BufT::with_capacity(alloc_len);
        heap_buf.as_uninit_slice_mut()
    };

    // For small inputs using quicksort is not yet beneficial, and a single
    // small-sort or two small-sorts plus a single merge outperforms it, so use
    // eager mode.
    let eager_sort = len <= T::small_sort_threshold() * 2;
    crate::slice::sort::stable::drift::sort(v, scratch, eager_sort, is_less);
}

#[doc(hidden)]
/// Abstracts owned memory buffer, so that sort code can live in core where no allocation is
/// possible. This trait can then be implemented in a place that has access to allocation.
pub trait BufGuard<T> {
    /// Creates new buffer that holds at least `capacity` memory.
    fn with_capacity(capacity: usize) -> Self;
    /// Returns mutable access to uninitialized memory owned by the buffer.
    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>];
}

#[repr(C)]
struct AlignedStorage<T, const N: usize> {
    _align: [T; 0],
    storage: [MaybeUninit<u8>; N],
}

impl<T, const N: usize> AlignedStorage<T, N> {
    fn new() -> Self {
        Self { _align: [], storage: [const { MaybeUninit::uninit() }; N] }
    }

    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let len = N / size_of::<T>();

        // SAFETY: `_align` ensures we are correctly aligned.
        unsafe { core::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), len) }
    }
}

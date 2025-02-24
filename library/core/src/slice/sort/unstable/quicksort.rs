//! This module contains an unstable quicksort and two partition implementations.

use crate::mem::{self, ManuallyDrop};
#[cfg(not(feature = "optimize_for_size"))]
use crate::slice::sort::shared::pivot::choose_pivot;
#[cfg(not(feature = "optimize_for_size"))]
use crate::slice::sort::shared::smallsort::UnstableSmallSortTypeImpl;
#[cfg(not(feature = "optimize_for_size"))]
use crate::slice::sort::unstable::heapsort;
use crate::{intrinsics, ptr};

/// Sorts `v` recursively.
///
/// If the slice had a predecessor in the original array, it is specified as `ancestor_pivot`.
///
/// `limit` is the number of allowed imbalanced partitions before switching to `heapsort`. If zero,
/// this function will immediately switch to heapsort.
#[cfg(not(feature = "optimize_for_size"))]
pub(crate) fn quicksort<'a, T, F>(
    mut v: &'a mut [T],
    mut ancestor_pivot: Option<&'a T>,
    mut limit: u32,
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
    loop {
        if v.len() <= T::small_sort_threshold() {
            T::small_sort(v, is_less);
            return;
        }

        // If too many bad pivot choices were made, simply fall back to heapsort in order to
        // guarantee `O(N x log(N))` worst-case.
        if limit == 0 {
            heapsort::heapsort(v, is_less);
            return;
        }

        limit -= 1;

        // Choose a pivot and try guessing whether the slice is already sorted.
        let pivot_pos = choose_pivot(v, is_less);

        // If the chosen pivot is equal to the predecessor, then it's the smallest element in the
        // slice. Partition the slice into elements equal to and elements greater than the pivot.
        // This case is usually hit when the slice contains many duplicate elements.
        if let Some(p) = ancestor_pivot {
            // SAFETY: We assume choose_pivot yields an in-bounds position.
            if !is_less(p, unsafe { v.get_unchecked(pivot_pos) }) {
                let num_lt = partition(v, pivot_pos, &mut |a, b| !is_less(b, a));

                // Continue sorting elements greater than the pivot. We know that `num_lt` contains
                // the pivot. So we can continue after `num_lt`.
                v = &mut v[(num_lt + 1)..];
                ancestor_pivot = None;
                continue;
            }
        }

        // Partition the slice.
        let num_lt = partition(v, pivot_pos, is_less);
        // SAFETY: partition ensures that `num_lt` will be in-bounds.
        unsafe { intrinsics::assume(num_lt < v.len()) };

        // Split the slice into `left`, `pivot`, and `right`.
        let (left, right) = v.split_at_mut(num_lt);
        let (pivot, right) = right.split_at_mut(1);
        let pivot = &pivot[0];

        // Recurse into the left side. We have a fixed recursion limit, testing shows no real
        // benefit for recursing into the shorter side.
        quicksort(left, ancestor_pivot, limit, is_less);

        // Continue with the right side.
        v = right;
        ancestor_pivot = Some(pivot);
    }
}

/// Takes the input slice `v` and re-arranges elements such that when the call returns normally
/// all elements that compare true for `is_less(elem, pivot)` where `pivot == v[pivot_pos]` are
/// on the left side of `v` followed by the other elements, notionally considered greater or
/// equal to `pivot`.
///
/// Returns the number of elements that are compared true for `is_less(elem, pivot)`.
///
/// If `is_less` does not implement a total order the resulting order and return value are
/// unspecified. All original elements will remain in `v` and any possible modifications via
/// interior mutability will be observable. Same is true if `is_less` panics or `v.len()`
/// exceeds `scratch.len()`.
pub(crate) fn partition<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    // Allows for panic-free code-gen by proving this property to the compiler.
    if len == 0 {
        return 0;
    }

    if pivot >= len {
        intrinsics::abort();
    }

    // SAFETY: We checked that `pivot` is in-bounds.
    unsafe {
        // Place the pivot at the beginning of slice.
        v.swap_unchecked(0, pivot);
    }
    let (pivot, v_without_pivot) = v.split_at_mut(1);

    // Assuming that Rust generates noalias LLVM IR we can be sure that a partition function
    // signature of the form `(v: &mut [T], pivot: &T)` guarantees that pivot and v can't alias.
    // Having this guarantee is crucial for optimizations. It's possible to copy the pivot value
    // into a stack value, but this creates issues for types with interior mutability mandating
    // a drop guard.
    let pivot = &mut pivot[0];

    // This construct is used to limit the LLVM IR generated, which saves large amounts of
    // compile-time by only instantiating the code that is needed. Idea by Frank Steffahn.
    let num_lt = (const { inst_partition::<T, F>() })(v_without_pivot, pivot, is_less);

    if num_lt >= len {
        intrinsics::abort();
    }

    // SAFETY: We checked that `num_lt` is in-bounds.
    unsafe {
        // Place the pivot between the two partitions.
        v.swap_unchecked(0, num_lt);
    }

    num_lt
}

const fn inst_partition<T, F: FnMut(&T, &T) -> bool>() -> fn(&mut [T], &T, &mut F) -> usize {
    const MAX_BRANCHLESS_PARTITION_SIZE: usize = 96;
    if mem::size_of::<T>() <= MAX_BRANCHLESS_PARTITION_SIZE {
        // Specialize for types that are relatively cheap to copy, where branchless optimizations
        // have large leverage e.g. `u64` and `String`.
        cfg_if! {
            if #[cfg(feature = "optimize_for_size")] {
                partition_lomuto_branchless_simple::<T, F>
            } else {
                partition_lomuto_branchless_cyclic::<T, F>
            }
        }
    } else {
        partition_hoare_branchy_cyclic::<T, F>
    }
}

/// See [`partition`].
fn partition_hoare_branchy_cyclic<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    if len == 0 {
        return 0;
    }

    // Optimized for large types that are expensive to move. Not optimized for integers. Optimized
    // for small code-gen, assuming that is_less is an expensive operation that generates
    // substantial amounts of code or a call. And that copying elements will likely be a call to
    // memcpy. Using 2 `ptr::copy_nonoverlapping` has the chance to be faster than
    // `ptr::swap_nonoverlapping` because `memcpy` can use wide SIMD based on runtime feature
    // detection. Benchmarks support this analysis.

    let mut gap_opt: Option<GapGuard<T>> = None;

    // SAFETY: The left-to-right scanning loop performs a bounds check, where we know that `left >=
    // v_base && left < right && right <= v_base.add(len)`. The right-to-left scanning loop performs
    // a bounds check ensuring that `right` is in-bounds. We checked that `len` is more than zero,
    // which means that unconditional `right = right.sub(1)` is safe to do. The exit check makes
    // sure that `left` and `right` never alias, making `ptr::copy_nonoverlapping` safe. The
    // drop-guard `gap` ensures that should `is_less` panic we always overwrite the duplicate in the
    // input. `gap.pos` stores the previous value of `right` and starts at `right` and so it too is
    // in-bounds. We never pass the saved `gap.value` to `is_less` while it is inside the `GapGuard`
    // thus any changes via interior mutability will be observed.
    unsafe {
        let v_base = v.as_mut_ptr();

        let mut left = v_base;
        let mut right = v_base.add(len);

        loop {
            // Find the first element greater than the pivot.
            while left < right && is_less(&*left, pivot) {
                left = left.add(1);
            }

            // Find the last element equal to the pivot.
            loop {
                right = right.sub(1);
                if left >= right || is_less(&*right, pivot) {
                    break;
                }
            }

            if left >= right {
                break;
            }

            // Swap the found pair of out-of-order elements via cyclic permutation.
            let is_first_swap_pair = gap_opt.is_none();

            if is_first_swap_pair {
                gap_opt = Some(GapGuard { pos: right, value: ManuallyDrop::new(ptr::read(left)) });
            }

            let gap = gap_opt.as_mut().unwrap_unchecked();

            // Single place where we instantiate ptr::copy_nonoverlapping in the partition.
            if !is_first_swap_pair {
                ptr::copy_nonoverlapping(left, gap.pos, 1);
            }
            gap.pos = right;
            ptr::copy_nonoverlapping(right, left, 1);

            left = left.add(1);
        }

        left.offset_from_unsigned(v_base)

        // `gap_opt` goes out of scope and overwrites the last wrong-side element on the right side
        // with the first wrong-side element of the left side that was initially overwritten by the
        // first wrong-side element on the right side element.
    }
}

#[cfg(not(feature = "optimize_for_size"))]
struct PartitionState<T> {
    // The current element that is being looked at, scans left to right through slice.
    right: *mut T,
    // Counts the number of elements that compared less-than, also works around:
    // https://github.com/rust-lang/rust/issues/117128
    num_lt: usize,
    // Gap guard that tracks the temporary duplicate in the input.
    gap: GapGuardRaw<T>,
}

#[cfg(not(feature = "optimize_for_size"))]
fn partition_lomuto_branchless_cyclic<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    // Novel partition implementation by Lukas Bergdoll and Orson Peters. Branchless Lomuto
    // partition paired with a cyclic permutation.
    // https://github.com/Voultapher/sort-research-rs/blob/main/writeup/lomcyc_partition/text.md

    let len = v.len();
    let v_base = v.as_mut_ptr();

    if len == 0 {
        return 0;
    }

    // SAFETY: We checked that `len` is more than zero, which means that reading `v_base` is safe to
    // do. From there we have a bounded loop where `v_base.add(i)` is guaranteed in-bounds. `v` and
    // `pivot` can't alias because of type system rules. The drop-guard `gap` ensures that should
    // `is_less` panic we always overwrite the duplicate in the input. `gap.pos` stores the previous
    // value of `right` and starts at `v_base` and so it too is in-bounds. Given `UNROLL_LEN == 2`
    // after the main loop we either have A) the last element in `v` that has not yet been processed
    // because `len % 2 != 0`, or B) all elements have been processed except the gap value that was
    // saved at the beginning with `ptr::read(v_base)`. In the case A) the loop will iterate twice,
    // first performing loop_body to take care of the last element that didn't fit into the unroll.
    // After that the behavior is the same as for B) where we use the saved value as `right` to
    // overwrite the duplicate. If this very last call to `is_less` panics the saved value will be
    // copied back including all possible changes via interior mutability. If `is_less` does not
    // panic and the code continues we overwrite the duplicate and do `right = right.add(1)`, this
    // is safe to do with `&mut *gap.value` because `T` is the same as `[T; 1]` and generating a
    // pointer one past the allocation is safe.
    unsafe {
        let mut loop_body = |state: &mut PartitionState<T>| {
            let right_is_lt = is_less(&*state.right, pivot);
            let left = v_base.add(state.num_lt);

            ptr::copy(left, state.gap.pos, 1);
            ptr::copy_nonoverlapping(state.right, left, 1);

            state.gap.pos = state.right;
            state.num_lt += right_is_lt as usize;

            state.right = state.right.add(1);
        };

        // Ideally we could just use GapGuard in PartitionState, but the reference that is
        // materialized with `&mut state` when calling `loop_body` would create a mutable reference
        // to the parent struct that contains the gap value, invalidating the reference pointer
        // created from a reference to the gap value in the cleanup loop. This is only an issue
        // under Stacked Borrows, Tree Borrows accepts the intuitive code using GapGuard as valid.
        let mut gap_value = ManuallyDrop::new(ptr::read(v_base));

        let mut state = PartitionState {
            num_lt: 0,
            right: v_base.add(1),

            gap: GapGuardRaw { pos: v_base, value: &mut *gap_value },
        };

        // Manual unrolling that works well on x86, Arm and with opt-level=s without murdering
        // compile-times. Leaving this to the compiler yields ok to bad results.
        let unroll_len = const { if mem::size_of::<T>() <= 16 { 2 } else { 1 } };

        let unroll_end = v_base.add(len - (unroll_len - 1));
        while state.right < unroll_end {
            if unroll_len == 2 {
                loop_body(&mut state);
                loop_body(&mut state);
            } else {
                loop_body(&mut state);
            }
        }

        // Single instantiate `loop_body` for both the unroll cleanup and cyclic permutation
        // cleanup. Optimizes binary-size and compile-time.
        let end = v_base.add(len);
        loop {
            let is_done = state.right == end;
            state.right = if is_done { state.gap.value } else { state.right };

            loop_body(&mut state);

            if is_done {
                mem::forget(state.gap);
                break;
            }
        }

        state.num_lt
    }
}

#[cfg(feature = "optimize_for_size")]
fn partition_lomuto_branchless_simple<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    pivot: &T,
    is_less: &mut F,
) -> usize {
    let mut left = 0;

    for right in 0..v.len() {
        // SAFETY: `left` can at max be incremented by 1 each loop iteration, which implies that
        // left <= right and that both are in-bounds.
        unsafe {
            let right_is_lt = is_less(v.get_unchecked(right), pivot);
            v.swap_unchecked(left, right);
            left += right_is_lt as usize;
        }
    }

    left
}

struct GapGuard<T> {
    pos: *mut T,
    value: ManuallyDrop<T>,
}

impl<T> Drop for GapGuard<T> {
    fn drop(&mut self) {
        // SAFETY: `self` MUST be constructed in a way that makes copying the gap value into
        // `self.pos` sound.
        unsafe {
            ptr::copy_nonoverlapping(&*self.value, self.pos, 1);
        }
    }
}

/// Ideally this wouldn't be needed and we could just use the regular GapGuard.
/// See comment in [`partition_lomuto_branchless_cyclic`].
#[cfg(not(feature = "optimize_for_size"))]
struct GapGuardRaw<T> {
    pos: *mut T,
    value: *mut T,
}

#[cfg(not(feature = "optimize_for_size"))]
impl<T> Drop for GapGuardRaw<T> {
    fn drop(&mut self) {
        // SAFETY: `self` MUST be constructed in a way that makes copying the gap value into
        // `self.pos` sound.
        unsafe {
            ptr::copy_nonoverlapping(self.value, self.pos, 1);
        }
    }
}

//! This module contains the hybrid top-level loop combining bottom-up Mergesort with top-down
//! Quicksort.

use crate::mem::MaybeUninit;
use crate::slice::sort::shared::find_existing_run;
use crate::slice::sort::shared::smallsort::StableSmallSortTypeImpl;
use crate::slice::sort::stable::merge::merge;
use crate::slice::sort::stable::quicksort::quicksort;
use crate::{cmp, intrinsics};

/// Sorts `v` based on comparison function `is_less`. If `eager_sort` is true,
/// it will only do small-sorts and physical merges, ensuring O(N * log(N))
/// worst-case complexity. `scratch.len()` must be at least
/// `max(v.len() - v.len() / 2, SMALL_SORT_GENERAL_SCRATCH_LEN)` otherwise the implementation may abort.
/// Fully ascending and descending inputs will be sorted with exactly N - 1
/// comparisons.
///
/// This is the main loop for driftsort, which uses powersort's heuristic to
/// determine in which order to merge runs, see below for details.
pub fn sort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    eager_sort: bool,
    is_less: &mut F,
) {
    let len = v.len();
    if len < 2 {
        return; // Removing this length check *increases* code size.
    }
    let scale_factor = merge_tree_scale_factor(len);

    // It's important to have a relatively high entry barrier for pre-sorted
    // runs, as the presence of a single such run will force on average several
    // merge operations and shrink the maximum quicksort size a lot. For that
    // reason we use sqrt(len) as our pre-sorted run threshold.
    const MIN_SQRT_RUN_LEN: usize = 64;
    let min_good_run_len = if len <= (MIN_SQRT_RUN_LEN * MIN_SQRT_RUN_LEN) {
        // For small input length `MIN_SQRT_RUN_LEN` would break pattern
        // detection of full or nearly sorted inputs.
        cmp::min(len - len / 2, MIN_SQRT_RUN_LEN)
    } else {
        sqrt_approx(len)
    };

    // (stack_len, runs, desired_depths) together form a stack maintaining run
    // information for the powersort heuristic. desired_depths[i] is the desired
    // depth of the merge node that merges runs[i] with the run that comes after
    // it.
    let mut stack_len = 0;
    let mut run_storage = MaybeUninit::<[DriftsortRun; 66]>::uninit();
    let runs: *mut DriftsortRun = run_storage.as_mut_ptr().cast();
    let mut desired_depth_storage = MaybeUninit::<[u8; 66]>::uninit();
    let desired_depths: *mut u8 = desired_depth_storage.as_mut_ptr().cast();

    let mut scan_idx = 0;
    let mut prev_run = DriftsortRun::new_sorted(0); // Initial dummy run.
    loop {
        // Compute the next run and the desired depth of the merge node between
        // prev_run and next_run. On the last iteration we create a dummy run
        // with root-level desired depth to fully collapse the merge tree.
        let (next_run, desired_depth);
        if scan_idx < len {
            next_run =
                create_run(&mut v[scan_idx..], scratch, min_good_run_len, eager_sort, is_less);
            desired_depth = merge_tree_depth(
                scan_idx - prev_run.len(),
                scan_idx,
                scan_idx + next_run.len(),
                scale_factor,
            );
        } else {
            next_run = DriftsortRun::new_sorted(0);
            desired_depth = 0;
        };

        // Process the merge nodes between earlier runs[i] that have a desire to
        // be deeper in the merge tree than the merge node for the splitpoint
        // between prev_run and next_run.
        //
        // SAFETY: first note that this is the only place we modify stack_len,
        // runs or desired depths. We maintain the following invariants:
        //  1. The first stack_len elements of runs/desired_depths are initialized.
        //  2. For all valid i > 0, desired_depths[i] < desired_depths[i+1].
        //  3. The sum of all valid runs[i].len() plus prev_run.len() equals
        //     scan_idx.
        unsafe {
            while stack_len > 1 && *desired_depths.add(stack_len - 1) >= desired_depth {
                // Desired depth greater than the upcoming desired depth, pop
                // left neighbor run from stack and merge into prev_run.
                let left = *runs.add(stack_len - 1);
                let merged_len = left.len() + prev_run.len();
                let merge_start_idx = scan_idx - merged_len;
                let merge_slice = v.get_unchecked_mut(merge_start_idx..scan_idx);
                prev_run = logical_merge(merge_slice, scratch, left, prev_run, is_less);
                stack_len -= 1;
            }

            // We now know that desired_depths[stack_len - 1] < desired_depth,
            // maintaining our invariant. This also guarantees we don't overflow
            // the stack as merge_tree_depth(..) <= 64 and thus we can only have
            // 64 distinct values on the stack before pushing, plus our initial
            // dummy run, while our capacity is 66.
            *runs.add(stack_len) = prev_run;
            *desired_depths.add(stack_len) = desired_depth;
            stack_len += 1;
        }

        // Break before overriding the last run with our dummy run.
        if scan_idx >= len {
            break;
        }

        scan_idx += next_run.len();
        prev_run = next_run;
    }

    if !prev_run.sorted() {
        stable_quicksort(v, scratch, is_less);
    }
}

// Nearly-Optimal Mergesorts: Fast, Practical Sorting Methods That Optimally
// Adapt to Existing Runs by J. Ian Munro and Sebastian Wild.
//
// This method forms a binary merge tree, where each internal node corresponds
// to a splitting point between the adjacent runs that have to be merged. If we
// visualize our array as the number line from 0 to 1, we want to find the
// dyadic fraction with smallest denominator that lies between the midpoints of
// our to-be-merged slices. The exponent in the dyadic fraction indicates the
// desired depth in the binary merge tree this internal node wishes to have.
// This does not always correspond to the actual depth due to the inherent
// imbalance in runs, but we follow it as closely as possible.
//
// As an optimization we rescale the number line from [0, 1) to [0, 2^62). Then
// finding the simplest dyadic fraction between midpoints corresponds to finding
// the most significant bit difference of the midpoints. We save scale_factor =
// ceil(2^62 / n) to perform this rescaling using a multiplication, avoiding
// having to repeatedly do integer divides. This rescaling isn't exact when n is
// not a power of two since we use integers and not reals, but the result is
// very close, and in fact when n < 2^30 the resulting tree is equivalent as the
// approximation errors stay entirely in the lower order bits.
//
// Thus for the splitting point between two adjacent slices [a, b) and [b, c)
// the desired depth of the corresponding merge node is CLZ((a+b)*f ^ (b+c)*f),
// where CLZ counts the number of leading zeros in an integer and f is our scale
// factor. Note that we omitted the division by two in the midpoint
// calculations, as this simply shifts the bits by one position (and thus always
// adds one to the result), and we only care about the relative depths.
//
// Finally, if we try to upper bound x = (a+b)*f giving x = (n-1 + n) * ceil(2^62 / n) then
//    x < (2^62 / n + 1) * 2n
//    x < 2^63 + 2n
// So as long as n < 2^62 we find that x < 2^64, meaning our operations do not
// overflow.
#[inline(always)]
fn merge_tree_scale_factor(n: usize) -> u64 {
    if usize::BITS > u64::BITS {
        panic!("Platform not supported");
    }

    ((1 << 62) + n as u64 - 1) / n as u64
}

// Note: merge_tree_depth output is < 64 when left < right as f*x and f*y must
// differ in some bit, and is <= 64 always.
#[inline(always)]
fn merge_tree_depth(left: usize, mid: usize, right: usize, scale_factor: u64) -> u8 {
    let x = left as u64 + mid as u64;
    let y = mid as u64 + right as u64;
    ((scale_factor * x) ^ (scale_factor * y)).leading_zeros() as u8
}

fn sqrt_approx(n: usize) -> usize {
    // Note that sqrt(n) = n^(1/2), and that 2^log2(n) = n. We combine these
    // two facts to approximate sqrt(n) as 2^(log2(n) / 2). Because our integer
    // log floors we want to add 0.5 to compensate for this on average, so our
    // initial approximation is 2^((1 + floor(log2(n))) / 2).
    //
    // We then apply an iteration of Newton's method to improve our
    // approximation, which for sqrt(n) is a1 = (a0 + n / a0) / 2.
    //
    // Finally we note that the exponentiation / division can be done directly
    // with shifts. We OR with 1 to avoid zero-checks in the integer log.
    let ilog = (n | 1).ilog2();
    let shift = (1 + ilog) / 2;
    ((1 << shift) + (n >> shift)) / 2
}

// Lazy logical runs as in Glidesort.
#[inline(always)]
fn logical_merge<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    left: DriftsortRun,
    right: DriftsortRun,
    is_less: &mut F,
) -> DriftsortRun {
    // If one or both of the runs are sorted do a physical merge, using
    // quicksort to sort the unsorted run if present. We also *need* to
    // physically merge if the combined runs would not fit in the scratch space
    // anymore (as this would mean we are no longer able to quicksort them).
    let len = v.len();
    let can_fit_in_scratch = len <= scratch.len();
    if !can_fit_in_scratch || left.sorted() || right.sorted() {
        if !left.sorted() {
            stable_quicksort(&mut v[..left.len()], scratch, is_less);
        }
        if !right.sorted() {
            stable_quicksort(&mut v[left.len()..], scratch, is_less);
        }
        merge(v, scratch, left.len(), is_less);

        DriftsortRun::new_sorted(len)
    } else {
        DriftsortRun::new_unsorted(len)
    }
}

/// Creates a new logical run.
///
/// A logical run can either be sorted or unsorted. If there is a pre-existing
/// run that clears the `min_good_run_len` threshold it is returned as a sorted
/// run. If not, the result depends on the value of `eager_sort`. If it is true,
/// then a sorted run of length `T::SMALL_SORT_THRESHOLD` is returned, and if it
/// is false an unsorted run of length `min_good_run_len` is returned.
fn create_run<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    min_good_run_len: usize,
    eager_sort: bool,
    is_less: &mut F,
) -> DriftsortRun {
    let len = v.len();
    if len >= min_good_run_len {
        let (run_len, was_reversed) = find_existing_run(v, is_less);

        // SAFETY: find_existing_run promises to return a valid run_len.
        unsafe { intrinsics::assume(run_len <= len) };

        if run_len >= min_good_run_len {
            if was_reversed {
                v[..run_len].reverse();
            }

            return DriftsortRun::new_sorted(run_len);
        }
    }

    if eager_sort {
        // We call quicksort with a len that will immediately call small-sort.
        // By not calling the small-sort directly here it can always be inlined into
        // the quicksort itself, making the recursive base case faster and is generally
        // more binary-size efficient.
        let eager_run_len = cmp::min(T::small_sort_threshold(), len);
        quicksort(&mut v[..eager_run_len], scratch, 0, None, is_less);
        DriftsortRun::new_sorted(eager_run_len)
    } else {
        DriftsortRun::new_unsorted(cmp::min(min_good_run_len, len))
    }
}

fn stable_quicksort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    is_less: &mut F,
) {
    // Limit the number of imbalanced partitions to `2 * floor(log2(len))`.
    // The binary OR by one is used to eliminate the zero-check in the logarithm.
    let limit = 2 * (v.len() | 1).ilog2();
    quicksort(v, scratch, limit, None, is_less);
}

/// Compactly stores the length of a run, and whether or not it is sorted. This
/// can always fit in a `usize` because the maximum slice length is [`isize::MAX`].
#[derive(Copy, Clone)]
struct DriftsortRun(usize);

impl DriftsortRun {
    #[inline(always)]
    fn new_sorted(length: usize) -> Self {
        Self((length << 1) | 1)
    }

    #[inline(always)]
    fn new_unsorted(length: usize) -> Self {
        Self(length << 1)
    }

    #[inline(always)]
    fn sorted(self) -> bool {
        self.0 & 1 == 1
    }

    #[inline(always)]
    fn len(self) -> usize {
        self.0 >> 1
    }
}

//! This module contains the implementation for `slice::select_nth_unstable`.
//! It uses an introselect algorithm based on ipnsort by Lukas Bergdoll and Orson Peters,
//! published at: <https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort>
//!
//! The fallback algorithm used for introselect is Median of Medians using Tukey's Ninther
//! for pivot selection. Using this as a fallback ensures O(n) worst case running time with
//! better performance than one would get using heapsort as fallback.

use crate::cfg_select;
use crate::mem::{self, SizedTypeProperties};
#[cfg(not(feature = "optimize_for_size"))]
use crate::slice::sort::shared::pivot::choose_pivot;
use crate::slice::sort::shared::smallsort::insertion_sort_shift_left;
use crate::slice::sort::unstable::quicksort::partition;

/// Reorders the slice such that the element at `index` is at its final sorted position.
pub(crate) fn partition_at_index<T, F>(
    v: &mut [T],
    index: usize,
    mut is_less: F,
) -> (&mut [T], &mut T, &mut [T])
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    // Puts a lower limit of 1 on `len`.
    if index >= len {
        panic!("partition_at_index index {} greater than length of slice {}", index, len);
    }

    if T::IS_ZST {
        // Sorting has no meaningful behavior on zero-sized types. Do nothing.
    } else if index == len - 1 {
        // Find max element and place it in the last position of the array. We're free to use
        // `unwrap()` here because we checked that `v` is not empty.
        let max_idx = max_index(v, &mut is_less).unwrap();
        v.swap(max_idx, index);
    } else if index == 0 {
        // Find min element and place it in the first position of the array. We're free to use
        // `unwrap()` here because we checked that `v` is not empty.
        let min_idx = min_index(v, &mut is_less).unwrap();
        v.swap(min_idx, index);
    } else {
        cfg_select! {
            feature = "optimize_for_size" => {
                median_of_medians(v, &mut is_less, index);
            }
            _ => {
                partition_at_index_loop(v, index, None, &mut is_less);
            }
        }
    }

    let (left, right) = v.split_at_mut(index);
    let (pivot, right) = right.split_at_mut(1);
    let pivot = &mut pivot[0];
    (left, pivot, right)
}

// For small sub-slices it's faster to use a dedicated small-sort, but because it is only called at
// most once, it doesn't make sense to use something more sophisticated than insertion-sort.
const INSERTION_SORT_THRESHOLD: usize = 16;

#[cfg(not(feature = "optimize_for_size"))]
fn partition_at_index_loop<'a, T, F>(
    mut v: &'a mut [T],
    mut index: usize,
    mut ancestor_pivot: Option<&'a T>,
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
    // Limit the amount of iterations and fall back to fast deterministic selection to ensure O(n)
    // worst case running time. This limit needs to be constant, because using `ilog2(len)` like in
    // `sort` would result in O(n log n) time complexity. The exact value of the limit is chosen
    // somewhat arbitrarily, but for most inputs bad pivot selections should be relatively rare, so
    // the limit is reached for sub-slices len / (2^limit or less). Which makes the remaining work
    // with the fallback minimal in relative terms.
    let mut limit = 16;

    loop {
        if v.len() <= INSERTION_SORT_THRESHOLD {
            if v.len() >= 2 {
                insertion_sort_shift_left(v, 1, is_less);
            }
            return;
        }

        if limit == 0 {
            median_of_medians(v, is_less, index);
            return;
        }

        limit -= 1;

        // Choose a pivot
        let pivot_pos = choose_pivot(v, is_less);

        // If the chosen pivot is equal to the predecessor, then it's the smallest element in the
        // slice. Partition the slice into elements equal to and elements greater than the pivot.
        // This case is usually hit when the slice contains many duplicate elements.
        if let Some(p) = ancestor_pivot {
            let pivot = &v[pivot_pos];

            if !is_less(p, pivot) {
                let num_lt = partition(v, pivot_pos, &mut |a, b| !is_less(b, a));

                // Continue sorting elements greater than the pivot. We know that `mid` contains
                // the pivot. So we can continue after `mid`.
                let mid = num_lt + 1;

                // If we've passed our index, then we're good.
                if mid > index {
                    return;
                }

                v = &mut v[mid..];
                index = index - mid;
                ancestor_pivot = None;
                continue;
            }
        }

        let mid = partition(v, pivot_pos, is_less);

        // Split the slice into `left`, `pivot`, and `right`.
        let (left, right) = v.split_at_mut(mid);
        let (pivot, right) = right.split_at_mut(1);
        let pivot = &pivot[0];

        if mid < index {
            v = right;
            index = index - mid - 1;
            ancestor_pivot = Some(pivot);
        } else if mid > index {
            v = left;
        } else {
            // If mid == index, then we're done, since partition() guaranteed that all elements
            // after mid are greater than or equal to mid.
            return;
        }
    }
}

/// Helper function that returns the index of the minimum element in the slice using the given
/// comparator function
fn min_index<T, F: FnMut(&T, &T) -> bool>(slice: &[T], is_less: &mut F) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .reduce(|acc, t| if is_less(t.1, acc.1) { t } else { acc })
        .map(|(i, _)| i)
}

/// Helper function that returns the index of the maximum element in the slice using the given
/// comparator function
fn max_index<T, F: FnMut(&T, &T) -> bool>(slice: &[T], is_less: &mut F) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .reduce(|acc, t| if is_less(acc.1, t.1) { t } else { acc })
        .map(|(i, _)| i)
}

/// Selection algorithm to select the k-th element from the slice in guaranteed O(n) time.
/// This is essentially a quickselect that uses Tukey's Ninther for pivot selection
fn median_of_medians<T, F: FnMut(&T, &T) -> bool>(mut v: &mut [T], is_less: &mut F, mut k: usize) {
    // Since this function isn't public, it should never be called with an out-of-bounds index.
    debug_assert!(k < v.len());

    // If T is as ZST, `partition_at_index` will already return early.
    debug_assert!(!T::IS_ZST);

    // We now know that `k < v.len() <= isize::MAX`
    loop {
        if v.len() <= INSERTION_SORT_THRESHOLD {
            if v.len() >= 2 {
                insertion_sort_shift_left(v, 1, is_less);
            }

            return;
        }

        // `median_of_{minima,maxima}` can't handle the extreme cases of the first/last element,
        // so we catch them here and just do a linear search.
        if k == v.len() - 1 {
            // Find max element and place it in the last position of the array. We're free to use
            // `unwrap()` here because we know v must not be empty.
            let max_idx = max_index(v, is_less).unwrap();
            v.swap(max_idx, k);
            return;
        } else if k == 0 {
            // Find min element and place it in the first position of the array. We're free to use
            // `unwrap()` here because we know v must not be empty.
            let min_idx = min_index(v, is_less).unwrap();
            v.swap(min_idx, k);
            return;
        }

        let p = median_of_ninthers(v, is_less);

        if p == k {
            return;
        } else if p > k {
            v = &mut v[..p];
        } else {
            // Since `p < k < v.len()`, `p + 1` doesn't overflow and is
            // a valid index into the slice.
            v = &mut v[p + 1..];
            k -= p + 1;
        }
    }
}

// Optimized for when `k` lies somewhere in the middle of the slice. Selects a pivot
// as close as possible to the median of the slice. For more details on how the algorithm
// operates, refer to the paper <https://drops.dagstuhl.de/opus/volltexte/2017/7612/pdf/LIPIcs-SEA-2017-24.pdf>.
fn median_of_ninthers<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], is_less: &mut F) -> usize {
    // use `saturating_mul` so the multiplication doesn't overflow on 16-bit platforms.
    let frac = if v.len() <= 1024 {
        v.len() / 12
    } else if v.len() <= 128_usize.saturating_mul(1024) {
        v.len() / 64
    } else {
        v.len() / 1024
    };

    let pivot = frac / 2;
    let lo = v.len() / 2 - pivot;
    let hi = frac + lo;
    let gap = (v.len() - 9 * frac) / 4;
    let mut a = lo - 4 * frac - gap;
    let mut b = hi + gap;
    for i in lo..hi {
        ninther(v, is_less, a, i - frac, b, a + 1, i, b + 1, a + 2, i + frac, b + 2);
        a += 3;
        b += 3;
    }

    median_of_medians(&mut v[lo..lo + frac], is_less, pivot);

    partition(v, lo + pivot, is_less)
}

/// Moves around the 9 elements at the indices a..i, such that
/// `v[d]` contains the median of the 9 elements and the other
/// elements are partitioned around it.
fn ninther<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    is_less: &mut F,
    a: usize,
    mut b: usize,
    c: usize,
    mut d: usize,
    e: usize,
    mut f: usize,
    g: usize,
    mut h: usize,
    i: usize,
) {
    b = median_idx(v, is_less, a, b, c);
    h = median_idx(v, is_less, g, h, i);
    if is_less(&v[h], &v[b]) {
        mem::swap(&mut b, &mut h);
    }
    if is_less(&v[f], &v[d]) {
        mem::swap(&mut d, &mut f);
    }
    if is_less(&v[e], &v[d]) {
        // do nothing
    } else if is_less(&v[f], &v[e]) {
        d = f;
    } else {
        if is_less(&v[e], &v[b]) {
            v.swap(e, b);
        } else if is_less(&v[h], &v[e]) {
            v.swap(e, h);
        }
        return;
    }
    if is_less(&v[d], &v[b]) {
        d = b;
    } else if is_less(&v[h], &v[d]) {
        d = h;
    }

    v.swap(d, e);
}

/// returns the index pointing to the median of the 3
/// elements `v[a]`, `v[b]` and `v[c]`
fn median_idx<T, F: FnMut(&T, &T) -> bool>(
    v: &[T],
    is_less: &mut F,
    mut a: usize,
    b: usize,
    mut c: usize,
) -> usize {
    if is_less(&v[c], &v[a]) {
        mem::swap(&mut a, &mut c);
    }
    if is_less(&v[c], &v[b]) {
        return c;
    }
    if is_less(&v[b], &v[a]) {
        return a;
    }
    b
}

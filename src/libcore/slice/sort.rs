// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Slice sorting
//!
//! This module contains an sort algorithm based on Orson Peters' pattern-defeating quicksort,
//! published at: https://github.com/orlp/pdqsort
//!
//! Unstable sorting is compatible with libcore because it doesn't allocate memory, unlike our
//! stable sorting implementation.

#![unstable(feature = "sort_unstable", issue = "40585")]

use cmp;
use mem;
use ptr;

/// Holds a value, but never drops it.
#[allow(unions_with_drop_fields)]
union NoDrop<T> {
    value: T
}

/// When dropped, copies from `src` into `dest`.
struct CopyOnDrop<T> {
    src: *mut T,
    dest: *mut T,
}

impl<T> Drop for CopyOnDrop<T> {
    fn drop(&mut self) {
        unsafe { ptr::copy_nonoverlapping(self.src, self.dest, 1); }
    }
}

/// Sorts a slice using insertion sort, which is `O(n^2)` worst-case.
fn insertion_sort<T, F>(v: &mut [T], is_less: &mut F)
    where F: FnMut(&T, &T) -> bool
{
    let len = v.len();

    for i in 1..len {
        unsafe {
            if is_less(v.get_unchecked(i), v.get_unchecked(i - 1)) {
                // There are three ways to implement insertion here:
                //
                // 1. Swap adjacent elements until the first one gets to its final destination.
                //    However, this way we copy data around more than is necessary. If elements are
                //    big structures (costly to copy), this method will be slow.
                //
                // 2. Iterate until the right place for the first element is found. Then shift the
                //    elements succeeding it to make room for it and finally place it into the
                //    remaining hole. This is a good method.
                //
                // 3. Copy the first element into a temporary variable. Iterate until the right
                //    place for it is found. As we go along, copy every traversed element into the
                //    slot preceding it. Finally, copy data from the temporary variable into the
                //    remaining hole. This method is very good. Benchmarks demonstrated slightly
                //    better performance than with the 2nd method.
                //
                // All methods were benchmarked, and the 3rd showed best results. So we chose that
                // one.
                let mut tmp = NoDrop { value: ptr::read(v.get_unchecked(i)) };

                // Intermediate state of the insertion process is always tracked by `hole`, which
                // serves two purposes:
                // 1. Protects integrity of `v` from panics in `is_less`.
                // 2. Fills the remaining hole in `v` in the end.
                //
                // Panic safety:
                //
                // If `is_less` panics at any point during the process, `hole` will get dropped and
                // fill the hole in `v` with `tmp`, thus ensuring that `v` still holds every object
                // it initially held exactly once.
                let mut hole = CopyOnDrop {
                    src: &mut tmp.value,
                    dest: v.get_unchecked_mut(i - 1),
                };
                ptr::copy_nonoverlapping(v.get_unchecked(i - 1), v.get_unchecked_mut(i), 1);

                for h in (0..i-1).rev() {
                    if !is_less(&tmp.value, v.get_unchecked(h)) {
                        break;
                    }
                    ptr::copy_nonoverlapping(v.get_unchecked(h), v.get_unchecked_mut(h + 1), 1);
                    hole.dest = v.get_unchecked_mut(h);
                }
                // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
            }
        }
    }
}

/// Sorts `v` using heapsort, which guarantees `O(n log n)` worst-case.
#[cold]
fn heapsort<T, F>(v: &mut [T], is_less: &mut F)
    where F: FnMut(&T, &T) -> bool
{
    // This binary heap respects the invariant `parent >= child`.
    let mut sift_down = |v: &mut [T], mut node| {
        loop {
            // Children of `node`:
            let left = 2 * node + 1;
            let right = 2 * node + 2;

            // Choose the greater child.
            let greater = if right < v.len() && is_less(&v[left], &v[right]) {
                right
            } else {
                left
            };

            // Stop if the invariant holds at `node`.
            if greater >= v.len() || !is_less(&v[node], &v[greater]) {
                break;
            }

            // Swap `node` with the greater child, move one step down, and continue sifting.
            v.swap(node, greater);
            node = greater;
        }
    };

    // Build the heap in linear time.
    for i in (0 .. v.len() / 2).rev() {
        sift_down(v, i);
    }

    // Pop maximal elements from the heap.
    for i in (1 .. v.len()).rev() {
        v.swap(0, i);
        sift_down(&mut v[..i], 0);
    }
}

/// Partitions `v` into elements smaller than `pivot`, followed by elements greater than or equal
/// to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
///
/// Partitioning is performed block-by-block in order to minimize the cost of branching operations.
/// This idea is presented in the [BlockQuicksort][pdf] paper.
///
/// [pdf]: http://drops.dagstuhl.de/opus/volltexte/2016/6389/pdf/LIPIcs-ESA-2016-38.pdf
fn partition_in_blocks<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
    where F: FnMut(&T, &T) -> bool
{
    // Number of elements in a typical block.
    const BLOCK: usize = 128;

    // The partitioning algorithm repeats the following steps until completion:
    //
    // 1. Trace a block from the left side to identify elements greater than or equal to the pivot.
    // 2. Trace a block from the right side to identify elements less than the pivot.
    // 3. Exchange the identified elements between the left and right side.
    //
    // We keep the following variables for a block of elements:
    //
    // 1. `block` - Number of elements in the block.
    // 2. `start` - Start pointer into the `offsets` array.
    // 3. `end` - End pointer into the `offsets` array.
    // 4. `offsets - Indices of out-of-order elements within the block.

    // The current block on the left side: `v[l .. l + block_l]`.
    let mut l = v.as_mut_ptr();
    let mut block_l = BLOCK;
    let mut start_l = ptr::null_mut();
    let mut end_l = ptr::null_mut();
    let mut offsets_l: [u8; BLOCK] = unsafe { mem::uninitialized() };

    // The current block on the right side: `v[r - block_r .. r]`.
    let mut r = unsafe { l.offset(v.len() as isize) };
    let mut block_r = BLOCK;
    let mut start_r = ptr::null_mut();
    let mut end_r = ptr::null_mut();
    let mut offsets_r: [u8; BLOCK] = unsafe { mem::uninitialized() };

    // Returns the number of elements between pointers `l` (inclusive) and `r` (exclusive).
    fn width<T>(l: *mut T, r: *mut T) -> usize {
        assert!(mem::size_of::<T>() > 0);
        (r as usize - l as usize) / mem::size_of::<T>()
    }

    loop {
        // We are done with partitioning block-by-block when `l` and `r` get very close. Then we do
        // some patch-up work in order to partition the remaining elements in between.
        let is_done = width(l, r) <= 2 * BLOCK;

        if is_done {
            // Number of remaining elements (still not compared to the pivot).
            let mut rem = width(l, r);
            if start_l < end_l || start_r < end_r {
                rem -= BLOCK;
            }

            // Adjust block sizes so that the left and right block don't overlap, but get perfectly
            // aligned to cover the whole remaining gap.
            if start_l < end_l {
                block_r = rem;
            } else if start_r < end_r {
                block_l = rem;
            } else {
                block_l = rem / 2;
                block_r = rem - block_l;
            }
            debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
            debug_assert!(width(l, r) == block_l + block_r);
        }

        if start_l == end_l {
            // Trace `block_l` elements from the left side.
            start_l = offsets_l.as_mut_ptr();
            end_l = offsets_l.as_mut_ptr();
            let mut elem = l;

            for i in 0..block_l {
                unsafe {
                    // Branchless comparison.
                    *end_l = i as u8;
                    end_l = end_l.offset(!is_less(&*elem, pivot) as isize);
                    elem = elem.offset(1);
                }
            }
        }

        if start_r == end_r {
            // Trace `block_r` elements from the right side.
            start_r = offsets_r.as_mut_ptr();
            end_r = offsets_r.as_mut_ptr();
            let mut elem = r;

            for i in 0..block_r {
                unsafe {
                    // Branchless comparison.
                    elem = elem.offset(-1);
                    *end_r = i as u8;
                    end_r = end_r.offset(is_less(&*elem, pivot) as isize);
                }
            }
        }

        // Number of out-of-order elements to swap between the left and right side.
        let count = cmp::min(width(start_l, end_l), width(start_r, end_r));

        if count > 0 {
            macro_rules! left { () => { l.offset(*start_l as isize) } }
            macro_rules! right { () => { r.offset(-(*start_r as isize) - 1) } }

            // Instead of swapping one pair at the time, it is more efficient to perform a cyclic
            // permutation. This is not strictly equivalent to swapping, but produces a similar
            // result using fewer memory operations.
            unsafe {
                let tmp = ptr::read(left!());
                ptr::copy_nonoverlapping(right!(), left!(), 1);

                for _ in 1..count {
                    start_l = start_l.offset(1);
                    ptr::copy_nonoverlapping(left!(), right!(), 1);
                    start_r = start_r.offset(1);
                    ptr::copy_nonoverlapping(right!(), left!(), 1);
                }

                ptr::copy_nonoverlapping(&tmp, right!(), 1);
                mem::forget(tmp);
                start_l = start_l.offset(1);
                start_r = start_r.offset(1);
            }
        }

        if start_l == end_l {
            // All out-of-order elements in the left block were moved. Move to the next block.
            l = unsafe { l.offset(block_l as isize) };
        }

        if start_r == end_r {
            // All out-of-order elements in the right block were moved. Move to the previous block.
            r = unsafe { r.offset(-(block_r as isize)) };
        }

        if is_done {
            break;
        }
    }

    // All that remains now is at most one block (either the left or the right) with out-of-order
    // elements that need to be moved. Such remaining elements can be simply shifted to the end
    // within their block.

    if start_l < end_l {
        // The left block remains.
        // Move it's remaining out-of-order elements to the far right.
        debug_assert_eq!(width(l, r), block_l);
        while start_l < end_l {
            unsafe {
                end_l = end_l.offset(-1);
                ptr::swap(l.offset(*end_l as isize), r.offset(-1));
                r = r.offset(-1);
            }
        }
        width(v.as_mut_ptr(), r)
    } else if start_r < end_r {
        // The right block remains.
        // Move it's remaining out-of-order elements to the far left.
        debug_assert_eq!(width(l, r), block_r);
        while start_r < end_r {
            unsafe {
                end_r = end_r.offset(-1);
                ptr::swap(l, r.offset(-(*end_r as isize) - 1));
                l = l.offset(1);
            }
        }
        width(v.as_mut_ptr(), l)
    } else {
        // Nothing else to do, we're done.
        width(v.as_mut_ptr(), l)
    }
}

/// Partitions `v` into elements smaller than `v[pivot]`, followed by elements greater than or
/// equal to `v[pivot]`.
///
/// Returns a tuple of:
///
/// 1. Number of elements smaller than `v[pivot]`.
/// 2. True if `v` was already partitioned.
fn partition<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> (usize, bool)
    where F: FnMut(&T, &T) -> bool
{
    let (mid, was_partitioned) = {
        // Place the pivot at the beginning of slice.
        v.swap(0, pivot);
        let (pivot, v) = v.split_at_mut(1);
        let pivot = &mut pivot[0];

        // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
        // operation panics, the pivot will be automatically written back into the slice.
        let mut tmp = NoDrop { value: unsafe { ptr::read(pivot) } };
        let _pivot_guard = CopyOnDrop {
            src: unsafe { &mut tmp.value },
            dest: pivot,
        };
        let pivot = unsafe { &tmp.value };

        // Find the first pair of out-of-order elements.
        let mut l = 0;
        let mut r = v.len();
        unsafe {
            // Find the first element greater then or equal to the pivot.
            while l < r && is_less(v.get_unchecked(l), pivot) {
                l += 1;
            }

            // Find the last element lesser that the pivot.
            while l < r && !is_less(v.get_unchecked(r - 1), pivot) {
                r -= 1;
            }
        }

        (l + partition_in_blocks(&mut v[l..r], pivot, is_less), l >= r)

        // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated
        // variable) back into the slice where it originally was. This step is critical in ensuring
        // safety!
    };

    // Place the pivot between the two partitions.
    v.swap(0, mid);

    (mid, was_partitioned)
}

/// Partitions `v` into elements equal to `v[pivot]` followed by elements greater than `v[pivot]`.
///
/// Returns the number of elements equal to the pivot. It is assumed that `v` does not contain
/// elements smaller than the pivot.
fn partition_equal<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> usize
    where F: FnMut(&T, &T) -> bool
{
    // Place the pivot at the beginning of slice.
    v.swap(0, pivot);
    let (pivot, v) = v.split_at_mut(1);
    let pivot = &mut pivot[0];

    // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
    // operation panics, the pivot will be automatically written back into the slice.
    let mut tmp = NoDrop { value: unsafe { ptr::read(pivot) } };
    let _pivot_guard = CopyOnDrop {
        src: unsafe { &mut tmp.value },
        dest: pivot,
    };
    let pivot = unsafe { &tmp.value };

    // Now partition the slice.
    let mut l = 0;
    let mut r = v.len();
    loop {
        unsafe {
            // Find the first element greater that the pivot.
            while l < r && !is_less(pivot, v.get_unchecked(l)) {
                l += 1;
            }

            // Find the last element equal to the pivot.
            while l < r && is_less(pivot, v.get_unchecked(r - 1)) {
                r -= 1;
            }

            // Are we done?
            if l >= r {
                break;
            }

            // Swap the found pair of out-of-order elements.
            r -= 1;
            ptr::swap(v.get_unchecked_mut(l), v.get_unchecked_mut(r));
            l += 1;
        }
    }

    // We found `l` elements equal to the pivot. Add 1 to account for the pivot itself.
    l + 1

    // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated variable)
    // back into the slice where it originally was. This step is critical in ensuring safety!
}

/// Scatters some elements around in an attempt to break patterns that might cause imbalanced
/// partitions in quicksort.
#[cold]
fn break_patterns<T>(v: &mut [T]) {
    let len = v.len();

    if len >= 8 {
        // A random number will be taken modulo this one. The modulus is a power of two so that we
        // can simply take bitwise "and", thus avoiding costly CPU operations.
        let modulus = (len / 4).next_power_of_two();
        debug_assert!(modulus >= 1 && modulus <= len / 2);

        // Pseudorandom number generation from the "Xorshift RNGs" paper by George Marsaglia.
        let mut random = len;
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random &= modulus - 1;
        debug_assert!(random < len / 2);

        // The first index.
        let a = len / 4 * 2;
        debug_assert!(a >= 1 && a < len - 2);

        // The second index.
        let b = len / 4 + random;
        debug_assert!(b >= 1 && b < len - 2);

        // Swap neighbourhoods of `a` and `b`.
        for i in 0..3 {
            v.swap(a - 1 + i, b - 1 + i);
        }
    }
}

/// Chooses a pivot in `v` and returns it's index.
///
/// Elements in `v` might be reordered in the process.
fn choose_pivot<T, F>(v: &mut [T], is_less: &mut F) -> usize
    where F: FnMut(&T, &T) -> bool
{
    // Minimal length to choose the median-of-medians method.
    // Shorter slices use the simple median-of-three method.
    const SHORTEST_MEDIAN_OF_MEDIANS: usize = 90;
    // Maximal number of swaps that can be performed in this function.
    const MAX_SWAPS: usize = 4 * 3;

    let len = v.len();

    // Three indices near which we are going to choose a pivot.
    let mut a = len / 4 * 1;
    let mut b = len / 4 * 2;
    let mut c = len / 4 * 3;

    // Counts the total number of swaps we are about to perform while sorting indices.
    let mut swaps = 0;

    if len >= 8 {
        // Swaps indices so that `v[a] <= v[b]`.
        let mut sort2 = |a: &mut usize, b: &mut usize| unsafe {
            if is_less(v.get_unchecked(*b), v.get_unchecked(*a)) {
                ptr::swap(a, b);
                swaps += 1;
            }
        };

        // Swaps indices so that `v[a] <= v[b] <= v[c]`.
        let mut sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
            sort2(a, b);
            sort2(b, c);
            sort2(a, b);
        };

        if len >= SHORTEST_MEDIAN_OF_MEDIANS {
            // Finds the median of `v[a - 1], v[a], v[a + 1]` and stores the index into `a`.
            let mut sort_adjacent = |a: &mut usize| {
                let tmp = *a;
                sort3(&mut (tmp - 1), a, &mut (tmp + 1));
            };

            // Find medians in the neighborhoods of `a`, `b`, and `c`.
            sort_adjacent(&mut a);
            sort_adjacent(&mut b);
            sort_adjacent(&mut c);
        }

        // Find the median among `a`, `b`, and `c`.
        sort3(&mut a, &mut b, &mut c);
    }

    if swaps < MAX_SWAPS {
        b
    } else {
        // The maximal number of swaps was performed. Chances are the slice is descending or mostly
        // descending, so reversing will probably help sort it faster.
        v.reverse();
        len - 1 - b
    }
}

/// Sorts `v` recursively.
///
/// If the slice had a predecessor in the original array, it is specified as `pred`.
///
/// `limit` is the number of allowed imbalanced partitions before switching to `heapsort`. If zero,
/// this function will immediately switch to heapsort.
fn recurse<'a, T, F>(mut v: &'a mut [T], is_less: &mut F, mut pred: Option<&'a T>, mut limit: usize)
    where F: FnMut(&T, &T) -> bool
{
    // Slices of up to this length get sorted using insertion sort.
    const MAX_INSERTION: usize = 16;

    // This is `true` if the last partitioning was balanced.
    let mut was_balanced = true;

    loop {
        let len = v.len();

        // Very short slices get sorted using insertion sort.
        if len <= MAX_INSERTION {
            insertion_sort(v, is_less);
            return;
        }

        // If too many bad pivot choices were made, simply fall back to heapsort in order to
        // guarantee `O(n log n)` worst-case.
        if limit == 0 {
            heapsort(v, is_less);
            return;
        }

        // If the last partitioning was imbalanced, try breaking patterns in the slice by shuffling
        // some elements around. Hopefully we'll choose a better pivot this time.
        if !was_balanced {
            break_patterns(v);
            limit -= 1;
        }

        let pivot = choose_pivot(v, is_less);

        // If the chosen pivot is equal to the predecessor, then it's the smallest element in the
        // slice. Partition the slice into elements equal to and elements greater than the pivot.
        // This case is usually hit when the slice contains many duplicate elements.
        if let Some(p) = pred {
            if !is_less(p, &v[pivot]) {
                let mid = partition_equal(v, pivot, is_less);

                // Continue sorting elements greater than the pivot.
                v = &mut {v}[mid..];
                continue;
            }
        }

        let (mid, was_partitioned) = partition(v, pivot, is_less);
        was_balanced = cmp::min(mid, len - mid) >= len / 8;

        // If the partitioning is decently balanced and the slice was already partitioned, there
        // are good chances it is also completely sorted. If so, we're done.
        if was_balanced && was_partitioned && v.windows(2).all(|w| !is_less(&w[1], &w[0])) {
            return;
        }

        // Split the slice into `left`, `pivot`, and `right`.
        let (left, right) = {v}.split_at_mut(mid);
        let (pivot, right) = right.split_at_mut(1);
        let pivot = &pivot[0];

        // Recurse into the shorter side only in order to minimize the total number of recursive
        // calls and consume less stack space. Then just continue with the longer side (this is
        // akin to tail recursion).
        if left.len() < right.len() {
            recurse(left, is_less, pred, limit);
            v = right;
            pred = Some(pivot);
        } else {
            recurse(right, is_less, Some(pivot), limit);
            v = left;
        }
    }
}

/// Sorts `v` using pattern-defeating quicksort, which is `O(n log n)` worst-case.
pub fn quicksort<T, F>(v: &mut [T], mut is_less: F)
    where F: FnMut(&T, &T) -> bool
{
    // Sorting has no meaningful behavior on zero-sized types.
    if mem::size_of::<T>() == 0 {
        return;
    }

    // Limit the number of imbalanced partitions to `floor(log2(len)) + 2`.
    let limit = mem::size_of::<usize>() * 8 - v.len().leading_zeros() as usize + 1;

    recurse(v, &mut is_less, None, limit);
}

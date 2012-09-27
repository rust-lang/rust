//! Sorting methods
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use vec::{len, push};
use core::cmp::{Eq, Ord};
use dvec::DVec;

export le;
export merge_sort;
export quick_sort;
export quick_sort3;
export Sort;

type Le<T> = pure fn(v1: &T, v2: &T) -> bool;

/**
 * Merge sort. Returns a new vector containing the sorted list.
 *
 * Has worst case O(n log n) performance, best case O(n), but
 * is not space efficient. This is a stable sort.
 */
fn merge_sort<T: Copy>(le: Le<T>, v: &[const T]) -> ~[T] {
    type Slice = (uint, uint);

    return merge_sort_(le, v, (0u, len(v)));

    fn merge_sort_<T: Copy>(le: Le<T>, v: &[const T], slice: Slice)
        -> ~[T] {
        let begin = slice.first();
        let end = slice.second();

        let v_len = end - begin;
        if v_len == 0u { return ~[]; }
        if v_len == 1u { return ~[v[begin]]; }

        let mid = v_len / 2u + begin;
        let a = (begin, mid);
        let b = (mid, end);
        return merge(le, merge_sort_(le, v, a), merge_sort_(le, v, b));
    }

    fn merge<T: Copy>(le: Le<T>, a: &[T], b: &[T]) -> ~[T] {
        let mut rs = vec::with_capacity(len(a) + len(b));
        let a_len = len(a);
        let mut a_ix = 0u;
        let b_len = len(b);
        let mut b_ix = 0u;
        while a_ix < a_len && b_ix < b_len {
            if le(&a[a_ix], &b[b_ix]) {
                vec::push(rs, a[a_ix]);
                a_ix += 1u;
            } else { vec::push(rs, b[b_ix]); b_ix += 1u; }
        }
        rs = vec::append(rs, vec::slice(a, a_ix, a_len));
        rs = vec::append(rs, vec::slice(b, b_ix, b_len));
        return rs;
    }
}

fn part<T: Copy>(compare_func: Le<T>, arr: &[mut T], left: uint,
                right: uint, pivot: uint) -> uint {
    let pivot_value = arr[pivot];
    arr[pivot] <-> arr[right];
    let mut storage_index: uint = left;
    let mut i: uint = left;
    while i < right {
        if compare_func(&arr[i], &pivot_value) {
            arr[i] <-> arr[storage_index];
            storage_index += 1u;
        }
        i += 1u;
    }
    arr[storage_index] <-> arr[right];
    return storage_index;
}

fn qsort<T: Copy>(compare_func: Le<T>, arr: &[mut T], left: uint,
             right: uint) {
    if right > left {
        let pivot = (left + right) / 2u;
        let new_pivot = part::<T>(compare_func, arr, left, right, pivot);
        if new_pivot != 0u {
            // Need to do this check before recursing due to overflow
            qsort::<T>(compare_func, arr, left, new_pivot - 1u);
        }
        qsort::<T>(compare_func, arr, new_pivot + 1u, right);
    }
}

/**
 * Quicksort. Sorts a mut vector in place.
 *
 * Has worst case O(n^2) performance, average case O(n log n).
 * This is an unstable sort.
 */
fn quick_sort<T: Copy>(compare_func: Le<T>, arr: &[mut T]) {
    if len::<T>(arr) == 0u { return; }
    qsort::<T>(compare_func, arr, 0u, len::<T>(arr) - 1u);
}

fn qsort3<T: Copy Ord Eq>(arr: &[mut T], left: int, right: int) {
    if right <= left { return; }
    let v: T = arr[right];
    let mut i: int = left - 1;
    let mut j: int = right;
    let mut p: int = i;
    let mut q: int = j;
    loop {
        i += 1;
        while arr[i] < v { i += 1; }
        j -= 1;
        while v < arr[j] {
            if j == left { break; }
            j -= 1;
        }
        if i >= j { break; }
        arr[i] <-> arr[j];
        if arr[i] == v {
            p += 1;
            arr[p] <-> arr[i];
        }
        if v == arr[j] {
            q -= 1;
            arr[j] <-> arr[q];
        }
    }
    arr[i] <-> arr[right];
    j = i - 1;
    i += 1;
    let mut k: int = left;
    while k < p {
        arr[k] <-> arr[j];
        k += 1;
        j -= 1;
        if k == len::<T>(arr) as int { break; }
    }
    k = right - 1;
    while k > q {
        arr[i] <-> arr[k];
        k -= 1;
        i += 1;
        if k == 0 { break; }
    }
    qsort3::<T>(arr, left, j);
    qsort3::<T>(arr, i, right);
}

/**
 * Fancy quicksort. Sorts a mut vector in place.
 *
 * Based on algorithm presented by ~[Sedgewick and Bentley]
 * (http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf).
 * According to these slides this is the algorithm of choice for
 * 'randomly ordered keys, abstract compare' & 'small number of key values'.
 *
 * This is an unstable sort.
 */
fn quick_sort3<T: Copy Ord Eq>(arr: &[mut T]) {
    if arr.len() <= 1 { return; }
    qsort3(arr, 0, (arr.len() - 1) as int);
}

trait Sort {
    fn qsort(self);
}

impl<T: Copy Ord Eq> &[mut T] : Sort {
    fn qsort(self) { quick_sort3(self); }
}

const MIN_MERGE: uint = 64;
const MIN_GALLOP: uint = 7;
const INITIAL_TMP_STORAGE: uint = 128;

fn timsort<T: Ord>(array: &[mut T]) {
    let size = array.len();
    if size < 2 {
        return;
    }

    if size < MIN_MERGE {
        let initRunLen = countRunAndMakeAscending(array);
        binarysort(array, initRunLen);
        return;
    }

    let ms = &MergeState();
    let minRun = minRunLength(size);

    let mut idx = 0;
    let mut remaining = size;
    loop {
        let arr = vec::mut_view(array, idx, size);
        let mut runLen: uint = countRunAndMakeAscending(arr);

        if runLen < minRun {
            let force = if remaining <= minRun {remaining} else {minRun};
            let slice = vec::mut_view(arr, 0, force);
            binarysort(slice, runLen);
            runLen = force;
        }

        ms.pushRun(idx, runLen);
        ms.mergeCollapse(array);

        idx += runLen;
        remaining -= runLen;
        if remaining == 0 { break; } 
    }
    
    ms.mergeForceCollapse(array);
}

fn binarysort<T: Ord>(array: &[mut T], start: uint) {
    let size = array.len();
    let mut start = start;
    assert start <= size;

    if start == 0 { start += 1; }

    let mut pivot = ~[];
    vec::reserve(&mut pivot, 1);
    unsafe { vec::raw::set_len(pivot, 1); };

    while start < size {
        unsafe {
            let tmpView = vec::mut_view(array, start, start+1);
            vec::raw::memmove(pivot, tmpView, 1);
        }
        let mut left = 0;
        let mut right = start;
        assert left <= right;

        while left < right {
            let mid = (left + right) >> 1;
            if pivot[0] < array[mid] {
                right = mid;
            } else {
                left = mid+1;
            }
        }
        assert left == right;
        let mut n = start-left;

        unsafe {
            moveVec(array, left+1, array, left, n);
        }
        array[left] <-> pivot[0];
        start += 1;
    }
    unsafe { vec::raw::set_len(pivot, 0); } // Forget the boxed element
}

/// Reverse the order of elements in a slice, in place
fn reverseSlice<T>(v: &[mut T], start: uint, end:uint) {
    let mut i = start;
    while i < end / 2 {
        v[i] <-> v[end - i - 1];
        i += 1;
    }
}

pure fn minRunLength(n: uint) -> uint {
    let mut n = n;
    let mut r = 0;   // becomes 1 if any 1 bits are shifted off

    while n >= MIN_MERGE {
        r |= n & 1;
        n >>= 1;
    }
    return n + r;
}

fn countRunAndMakeAscending<T: Ord>(array: &[mut T]) -> uint {
    let size = array.len();
    assert size > 0;
    if size == 1 { return 1; }

    let mut run = 2;
    if array[1] < array[0] {
        while run < size && array[run] < array[run-1] {
            run += 1;
        }
        reverseSlice(array, 0, run);
    } else {
        while run < size && array[run] >= array[run-1] {
            run += 1;
        }
    }
    
    return run;
}

pure fn gallopLeft<T: Ord>(key: &const T, array: &[const T], hint: uint) -> uint {  
    let size = array.len();
    assert size != 0 && hint < size;

    let mut lastOfs = 0;
    let mut ofs = 1;
    
    if *key > array[hint] {
        // Gallop right until array[hint+lastOfs] < key <= array[hint+ofs]
        let maxOfs = size - hint;
        while ofs < maxOfs && *key > array[hint+ofs] {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if ofs < lastOfs { ofs = maxOfs; } // uint overflow guard
        }
        if ofs > maxOfs { ofs = maxOfs; }

        lastOfs += hint;
        ofs += hint;
    } else {
        let maxOfs = hint + 1;
        while ofs < maxOfs && *key <= array[hint-ofs] {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if ofs < lastOfs { ofs = maxOfs; } // uint overflow guard
        }
        
        if ofs > maxOfs { ofs = maxOfs; }

        let tmp = lastOfs;
        lastOfs = hint - ofs;
        ofs = hint - tmp;
    }
    assert (lastOfs < ofs || lastOfs+1 < ofs+1) && ofs <= size;

    lastOfs += 1;
    while lastOfs < ofs {
        let m = lastOfs + ((ofs - lastOfs) >> 1);
        if *key > array[m] {
            lastOfs = m+1;
        } else {
            ofs = m;
        }
    }
    assert lastOfs == ofs;
    return ofs;
}

pure fn gallopRight<T: Ord>(key: &const T, array: &[const T], hint: uint) -> uint {
    let size = array.len();
    assert size != 0 && hint < size;

    let mut lastOfs = 0;
    let mut ofs = 1;
    
    if *key >= array[hint] {
        // Gallop right until array[hint+lastOfs] <= key < array[hint+ofs]
        let maxOfs = size - hint;
        while ofs < maxOfs && *key >= array[hint+ofs] {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if ofs < lastOfs { ofs = maxOfs; }
        }
        if ofs > maxOfs { ofs = maxOfs; }

        lastOfs += hint;
        ofs += hint;    
    } else {
        // Gallop left until array[hint-ofs] <= key < array[hint-lastOfs]
        let maxOfs = hint + 1;
        while ofs < maxOfs && *key < array[hint-ofs] {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if ofs < lastOfs { ofs = maxOfs; }
        }
        if ofs > maxOfs { ofs = maxOfs; }

        let tmp = lastOfs;
        lastOfs = hint - ofs;
        ofs = hint - tmp;
    }

    assert (lastOfs < ofs || lastOfs+1 < ofs+1) && ofs <= size;

    lastOfs += 1;
    while lastOfs < ofs {
        let m = lastOfs + ((ofs - lastOfs) >> 1);

        if *key >= array[m] {
            lastOfs = m + 1;
        } else {
            ofs = m;
        }
    }
    assert lastOfs == ofs;
    return ofs;
}

struct RunState {
    base: uint,
    len: uint,
}

struct MergeState<T> {
    mut minGallop: uint,
    mut tmp: ~[T],
    runs: DVec<RunState>,

    drop {
        unsafe {
            vec::raw::set_len(self.tmp, 0);
        }
    }
}

fn MergeState<T>() -> MergeState<T> {
    let mut tmp = ~[];
    vec::reserve(&mut tmp, INITIAL_TMP_STORAGE); 
    MergeState {
        minGallop: MIN_GALLOP,
        tmp: tmp,
        runs: DVec(),
    }
}

impl<T: Ord> &MergeState<T> {
    fn pushRun(runBase: uint, runLen: uint) {
        let tmp = RunState{base: runBase, len: runLen};
        self.runs.push(tmp);
    }

    fn mergeAt(n: uint, array: &[mut T]) {
        let mut size = self.runs.len();
        assert size >= 2;
        assert n == size-2 || n == size-3;

        do self.runs.borrow_mut |arr| {

            let mut b1 = arr[n].base;
            let mut l1 = arr[n].len;
            let b2 = arr[n+1].base;
            let l2 = arr[n+1].len;

            assert l1 > 0 && l2 > 0;
            assert b1 + l1 == b2;

            arr[n].len = l1 + l2;
            if n == size-3 {
                arr[n+1].base = arr[n+2].base;
                arr[n+1].len = arr[n+2].len;
            }

            let slice = vec::mut_view(array, b1, b1+l1);
            let k = gallopRight(&const array[b2], slice, 0);
            b1 += k;
            l1 -= k;
            if l1 != 0 {
                let slice = vec::mut_view(array, b2, b2+l2);
                let l2 = gallopLeft(
                    &const array[b1+l1-1],slice,l2-1);
                if l2 > 0 {
                    if l1 <= l2 {
                        self.mergeLo(array, b1, l1, b2, l2);
                    } else {
                        self.mergeHi(array, b1, l1, b2, l2);
                    }
                }
            }
        }
        self.runs.pop();
    }

    fn mergeLo(array: &[mut T], base1: uint, len1: uint, base2: uint, len2: uint) {
        assert len1 != 0 && len2 != 0 && base1+len1 == base2;
        
        vec::reserve(&mut self.tmp, len1);

        unsafe {
            vec::raw::set_len(self.tmp, len1);
            moveVec(self.tmp, 0, array, base1, len1);
        }

        let mut c1 = 0;
        let mut c2 = base2;
        let mut dest = base1;
        let mut len1 = len1;
        let mut len2 = len2;

        array[dest] <-> array[c2];
        dest += 1; c2 += 1; len2 -= 1;

        if len2 == 0 {
            unsafe {
                moveVec(array, dest, self.tmp, 0, len1);
                vec::raw::set_len(self.tmp, 0); // Forget the elements
            }
            return;
        }
        if len1 == 1 {
            unsafe {
                moveVec(array, dest, array, c2, len2);
                array[dest+len2] <-> self.tmp[c1];
                vec::raw::set_len(self.tmp, 0); // Forget the element
            }
            return;
        }

        let mut minGallop = self.minGallop;
        loop {
            let mut count1 = 0;
            let mut count2 = 0;
            let mut breakOuter = false;

            loop {
                assert len1 > 1 && len2 != 0;
                if array[c2] < self.tmp[c1] {
                    array[dest] <-> array[c2];
                    dest += 1; c2 += 1; len2 -= 1;
                    count2 += 1; count1 = 0;
                    if len2 == 0 {
                        breakOuter = true;
                    }
                } else {
                    array[dest] <-> self.tmp[c1];
                    dest += 1; c1 += 1; len1 -= 1;
                    count1 += 1; count2 = 0;
                    if len1 == 1 {
                        breakOuter = true;
                    }
                }
                if breakOuter || ((count1 | count2) >= minGallop) {
                    break;
                }
            }
            if breakOuter { break; }

            // Start to gallop
            loop {
                assert len1 > 1 && len2 != 0;

                let tmpView = vec::mut_view(self.tmp, c1, c1+len1);
                count1 = gallopRight(&const array[c2], tmpView, 0);
                if count1 != 0 {
                    unsafe {
                        moveVec(array, dest, self.tmp, c1, count1);
                    }
                    dest += count1; c1 += count1; len1 -= count1;
                    if len1 <= 1 { breakOuter = true; break; }
                }
                array[dest] <-> array[c2];
                dest += 1; c2 += 1; len2 -= 1;
                if len2 == 0 { breakOuter = true; break; }

                let tmpView = vec::mut_view(array, c2, c2+len2);
                count2 = gallopLeft(&const self.tmp[c1], tmpView, 0);
                if count2 != 0 {
                    unsafe {
                        moveVec(array, dest, array, c2, count2);
                    }
                    dest += count2; c2 += count2; len2 -= count2;
                    if len2 == 0 { breakOuter = true; break; }
                }
                array[dest] <-> self.tmp[c1];
                dest += 1; c1 += 1; len1 -= 1;
                if len1 == 1 { breakOuter = true; break; }
                minGallop -= 1;
                if !(count1 >= MIN_GALLOP || count2 >= MIN_GALLOP) { break; } 
            }
            if breakOuter { break; }
            if minGallop < 0 { minGallop = 0; }
            minGallop += 2; // Penalize for leaving gallop
        }
        self.minGallop = if minGallop < 1 { 1 } else { minGallop };

        if len1 == 1 {
            assert len2 > 0;
            unsafe {
                moveVec(array, dest, array, c2, len2);
            }
            array[dest+len2] <-> self.tmp[c1];
        } else if len1 == 0 {
            fail fmt!("Method mergeLo violates its contract! %?", len1);
        } else {
            assert len2 == 0;
            assert len1 > 1;
            unsafe {
                moveVec(array, dest, self.tmp, c1, len1);
            }
        }
        unsafe { vec::raw::set_len(self.tmp, 0); }
    }

    fn mergeHi(array: &[mut T], base1: uint, len1: uint, base2: uint, len2: uint) {
        assert len1 != 1 && len2 != 0 && base1 + len1 == base2;

        vec::reserve(&mut self.tmp, len2);

        unsafe {
            vec::raw::set_len(self.tmp, len2);
            moveVec(self.tmp, 0, array, base2, len2);
        }

        let mut c1 = base1 + len1 - 1;
        let mut c2 = len2 - 1;
        let mut dest = base2 + len2 - 1;
        let mut len1 = len1;
        let mut len2 = len2;

        array[dest] <-> array[c1];
        dest -= 1; c1 -= 1; len1 -= 1;

        if len1 == 0 {
            unsafe {
                moveVec(array, dest-(len2-1), self.tmp, 0, len2);
                vec::raw::set_len(self.tmp, 0); // Forget the elements
            }
            return;
        }
        if len2 == 1 {
            dest -= len1;
            c1 -= len1;
            unsafe {
                moveVec(array, dest+1, array, c1+1, len1);
                array[dest] <-> self.tmp[c2];
                vec::raw::set_len(self.tmp, 0); // Forget the element
            }
            return;
        }

        let mut minGallop = self.minGallop;
        loop {
            let mut count1 = 0;
            let mut count2 = 0;
            let mut breakOuter = false;

            loop {
                assert len1 != 0 && len2 > 1;
                if self.tmp[c2] < array[c1] {
                    array[dest] <-> array[c1];
                    dest -= 1; c1 -= 1; len1 -= 1;
                    count1 += 1; count2 = 0;
                    if len1 == 0 {
                        breakOuter = true;
                    }
                } else {
                    array[dest] <-> self.tmp[c2];
                    dest -= 1; c2 -= 1; len2 -= 1;
                    count2 += 1; count1 = 0;
                    if len2 == 1 {
                        breakOuter = true;
                    }
                }
                if breakOuter || ((count1 | count2) >= minGallop) {
                    break;
                }
            }
            if breakOuter { break; }
    
            // Start to gallop
            loop {
                assert len2 > 1 && len1 != 0;

                let tmpView = vec::mut_view(array, base1, base1+len1);
                count1 = len1-gallopRight(&const self.tmp[c2], tmpView, len1-1);

                if count1 != 0 {
                    dest -= count1; c1 -= count1; len1 -= count1;
                    unsafe {
                        moveVec(array, dest+1, array, c1+1, count1);
                    }
                    if len1 == 0 { breakOuter = true; break; }
                }

                array[dest] <-> self.tmp[c2];
                dest -= 1; c2 -= 1; len2 -= 1;
                if len2 == 1 { breakOuter = true; break; }

                let tmpView = vec::mut_view(self.tmp, 0, len2);
                let gL = gallopLeft(&const array[c1], tmpView, len2-1);
                count2 = len2 - gL;
                if count2 != 0 {
                    dest -= count2; c2 -= count2; len2 -= count2;
                    unsafe {
                        moveVec(array, dest+1, self.tmp, c2+1, count2);
                    }
                    if len2 <= 1 { breakOuter = true; break; }
                }
                array[dest] <-> array[c1];
                dest -= 1; c1 -= 1; len1 -= 1;
                if len1 == 0 { breakOuter = true; break; }
                minGallop -= 1;
                if !(count1 >= MIN_GALLOP || count2 >= MIN_GALLOP) { break; } 
            }
            
            if breakOuter { break; }
            if minGallop < 0 { minGallop = 0; }
            minGallop += 2; // Penalize for leaving gallop
        }
        self.minGallop = if minGallop < 1 { 1 } else { minGallop };
        
        if len2 == 1 {
            assert len1 > 0;
            dest -= len1;
            c1 -= len1;
            unsafe {
                moveVec(array, dest+1, array, c1+1, len1);
            }
            array[dest] <-> self.tmp[c2];
        } else if len2 == 0 {
            fail fmt!("Method mergeHi violates its contract! %?", len2);
        } else {
            assert len1 == 0;
            assert len2 != 0;
            unsafe {
                moveVec(array, dest-(len2-1), self.tmp, 0, len2);
            }
        }
        unsafe { vec::raw::set_len(self.tmp, 0); }
    }

    fn mergeCollapse(array: &[mut T]) {
        while self.runs.len() > 1 {
            let mut n = self.runs.len()-2;
            let chk = do self.runs.borrow |arr| {
                if n > 0 && arr[n-1].len <= arr[n].len + arr[n+1].len {
                    if arr[n-1].len < arr[n+1].len { n -= 1; }
                    true
                } else if arr[n].len <= arr[n+1].len {
                    true
                } else {
                    false
                }
            };
            if !chk { break; }
            self.mergeAt(n, array);
        }
    }

    fn mergeForceCollapse(array: &[mut T]) {
        while self.runs.len() > 1 {
            let mut n = self.runs.len()-2;
            if n > 0 {
                do self.runs.borrow |arr| {
                    if arr[n-1].len < arr[n+1].len {
                        n -= 1;
                    }
                }
            }
            self.mergeAt(n, array);
        }
    }
}

// Moves elements to from dest to from
// Unsafe as it makes the from parameter invalid between s2 and s2+len
#[inline(always)]
unsafe fn moveVec<T>(dest: &[mut T], s1: uint, from: &[const T], s2: uint, len: uint) {   
    assert s1+len <= dest.len() && s2+len <= from.len();

    do vec::as_mut_buf(dest) |p, _len| {
        let destPtr = ptr::mut_offset(p, s1);

        do vec::as_const_buf(from) |p, _len| {
            let fromPtr = ptr::const_offset(p, s2);

            ptr::memmove(destPtr, fromPtr, len);
        }
    }
}

#[cfg(test)]
mod test_qsort3 {
    #[legacy_exports];
    fn check_sort(v1: &[mut int], v2: &[mut int]) {
        let len = vec::len::<int>(v1);
        quick_sort3::<int>(v1);
        let mut i = 0u;
        while i < len {
            log(debug, v2[i]);
            assert (v2[i] == v1[i]);
            i += 1u;
        }
    }

    #[test]
    fn test() {
        {
            let v1 = ~[mut 3, 7, 4, 5, 2, 9, 5, 8];
            let v2 = ~[mut 2, 3, 4, 5, 5, 7, 8, 9];
            check_sort(v1, v2);
        }
        {
            let v1 = ~[mut 1, 1, 1];
            let v2 = ~[mut 1, 1, 1];
            check_sort(v1, v2);
        }
        {
            let v1: ~[mut int] = ~[mut];
            let v2: ~[mut int] = ~[mut];
            check_sort(v1, v2);
        }
        { let v1 = ~[mut 9]; let v2 = ~[mut 9]; check_sort(v1, v2); }
        {
            let v1 = ~[mut 9, 3, 3, 3, 9];
            let v2 = ~[mut 3, 3, 3, 9, 9];
            check_sort(v1, v2);
        }
    }
}

#[cfg(test)]
mod test_qsort {
    #[legacy_exports];
    fn check_sort(v1: &[mut int], v2: &[mut int]) {
        let len = vec::len::<int>(v1);
        pure fn leual(a: &int, b: &int) -> bool { *a <= *b }
        quick_sort::<int>(leual, v1);
        let mut i = 0u;
        while i < len {
            log(debug, v2[i]);
            assert (v2[i] == v1[i]);
            i += 1u;
        }
    }

    #[test]
    fn test() {
        {
            let v1 = ~[mut 3, 7, 4, 5, 2, 9, 5, 8];
            let v2 = ~[mut 2, 3, 4, 5, 5, 7, 8, 9];
            check_sort(v1, v2);
        }
        {
            let v1 = ~[mut 1, 1, 1];
            let v2 = ~[mut 1, 1, 1];
            check_sort(v1, v2);
        }
        {
            let v1: ~[mut int] = ~[mut];
            let v2: ~[mut int] = ~[mut];
            check_sort(v1, v2);
        }
        { let v1 = ~[mut 9]; let v2 = ~[mut 9]; check_sort(v1, v2); }
        {
            let v1 = ~[mut 9, 3, 3, 3, 9];
            let v2 = ~[mut 3, 3, 3, 9, 9];
            check_sort(v1, v2);
        }
    }

    // Regression test for #750
    #[test]
    fn test_simple() {
        let names = ~[mut 2, 1, 3];

        let expected = ~[1, 2, 3];

        sort::quick_sort(|x, y| { int::le(*x, *y) }, names);

        let immut_names = vec::from_mut(names);

        let pairs = vec::zip(expected, immut_names);
        for vec::each(pairs) |p| {
            let (a, b) = *p;
            debug!("%d %d", a, b);
            assert (a == b);
        }
    }
}

#[cfg(test)]
mod tests {
    #[legacy_exports];

    fn check_sort(v1: &[int], v2: &[int]) {
        let len = vec::len::<int>(v1);
        pure fn le(a: &int, b: &int) -> bool { *a <= *b }
        let f = le;
        let v3 = merge_sort::<int>(f, v1);
        let mut i = 0u;
        while i < len {
            log(debug, v3[i]);
            assert (v3[i] == v2[i]);
            i += 1u;
        }
    }

    #[test]
    fn test() {
        {
            let v1 = ~[3, 7, 4, 5, 2, 9, 5, 8];
            let v2 = ~[2, 3, 4, 5, 5, 7, 8, 9];
            check_sort(v1, v2);
        }
        { let v1 = ~[1, 1, 1]; let v2 = ~[1, 1, 1]; check_sort(v1, v2); }
        { let v1:~[int] = ~[]; let v2:~[int] = ~[]; check_sort(v1, v2); }
        { let v1 = ~[9]; let v2 = ~[9]; check_sort(v1, v2); }
        {
            let v1 = ~[9, 3, 3, 3, 9];
            let v2 = ~[3, 3, 3, 9, 9];
            check_sort(v1, v2);
        }
    }

    #[test]
    fn test_merge_sort_mutable() {
        pure fn le(a: &int, b: &int) -> bool { *a <= *b }
        let v1 = ~[mut 3, 2, 1];
        let v2 = merge_sort(le, v1);
        assert v2 == ~[1, 2, 3];
    }

    #[test]
    fn test_merge_sort_stability()
    {
        // tjc: funny that we have to use parens
        pure fn ile(x: &(&static/str), y: &(&static/str)) -> bool
        {
            unsafe            // to_lower is not pure...
            {
                let x = x.to_lower();
                let y = y.to_lower();
                x <= y
            }
        }

        let names1 = ~["joe bob", "Joe Bob", "Jack Brown", "JOE Bob",
                       "Sally Mae", "JOE BOB", "Alex Andy"];
        let names2 = ~["Alex Andy", "Jack Brown", "joe bob", "Joe Bob",
                       "JOE Bob", "JOE BOB", "Sally Mae"];
        let names3 = merge_sort(ile, names1);
        assert names3 == names2;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

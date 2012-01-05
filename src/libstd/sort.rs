/*
Module: sort

Sorting methods
*/
import vec::{len, slice};

export merge_sort;
export quick_sort;
export quick_sort3;

/* Type: lteq */
type lteq<T> = block(T, T) -> bool;

/*
Function: merge_sort

Merge sort. Returns a new vector containing the sorted list.

Has worst case O(n log n) performance, best case O(n), but
is not space efficient. This is a stable sort.
*/
fn merge_sort<T: copy>(le: lteq<T>, v: [const T]) -> [T] {
    fn merge<T: copy>(le: lteq<T>, a: [T], b: [T]) -> [T] {
        let rs: [T] = [];
        let a_len: uint = len::<T>(a);
        let a_ix: uint = 0u;
        let b_len: uint = len::<T>(b);
        let b_ix: uint = 0u;
        while a_ix < a_len && b_ix < b_len {
            if le(a[a_ix], b[b_ix]) {
                rs += [a[a_ix]];
                a_ix += 1u;
            } else { rs += [b[b_ix]]; b_ix += 1u; }
        }
        rs += slice::<T>(a, a_ix, a_len);
        rs += slice::<T>(b, b_ix, b_len);
        ret rs;
    }
    let v_len: uint = len::<T>(v);
    if v_len == 0u { ret []; }
    if v_len == 1u { ret [v[0]]; }
    let mid: uint = v_len / 2u;
    let a: [T] = slice::<T>(v, 0u, mid);
    let b: [T] = slice::<T>(v, mid, v_len);
    ret merge::<T>(le, merge_sort::<T>(le, a), merge_sort::<T>(le, b));
}

fn part<T: copy>(compare_func: lteq<T>, arr: [mutable T], left: uint,
                right: uint, pivot: uint) -> uint {
    let pivot_value = arr[pivot];
    arr[pivot] <-> arr[right];
    let storage_index: uint = left;
    let i: uint = left;
    while i < right {
        if compare_func(copy arr[i], pivot_value) {
            arr[i] <-> arr[storage_index];
            storage_index += 1u;
        }
        i += 1u;
    }
    arr[storage_index] <-> arr[right];
    ret storage_index;
}

fn qsort<T: copy>(compare_func: lteq<T>, arr: [mutable T], left: uint,
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

/*
Function: quick_sort

Quicksort. Sorts a mutable vector in place.

Has worst case O(n^2) performance, average case O(n log n).
This is an unstable sort.
*/
fn quick_sort<T: copy>(compare_func: lteq<T>, arr: [mutable T]) {
    if len::<T>(arr) == 0u { ret; }
    qsort::<T>(compare_func, arr, 0u, len::<T>(arr) - 1u);
}

fn qsort3<T: copy>(compare_func_lt: lteq<T>, compare_func_eq: lteq<T>,
                  arr: [mutable T], left: int, right: int) {
    if right <= left { ret; }
    let v: T = arr[right];
    let i: int = left - 1;
    let j: int = right;
    let p: int = i;
    let q: int = j;
    while true {
        i += 1;
        while compare_func_lt(copy arr[i], v) { i += 1; }
        j -= 1;
        while compare_func_lt(v, copy arr[j]) {
            if j == left { break; }
            j -= 1;
        }
        if i >= j { break; }
        arr[i] <-> arr[j];
        if compare_func_eq(copy arr[i], v) {
            p += 1;
            arr[p] <-> arr[i];
        }
        if compare_func_eq(v, copy arr[j]) {
            q -= 1;
            arr[j] <-> arr[q];
        }
    }
    arr[i] <-> arr[right];
    j = i - 1;
    i += 1;
    let k: int = left;
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
    qsort3::<T>(compare_func_lt, compare_func_eq, arr, left, j);
    qsort3::<T>(compare_func_lt, compare_func_eq, arr, i, right);
}

// FIXME: This should take lt and eq types
/*
Function: quick_sort3

Fancy quicksort. Sorts a mutable vector in place.

Based on algorithm presented by Sedgewick and Bentley
<http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf>.
According to these slides this is the algorithm of choice for
'randomly ordered keys, abstract compare' & 'small number of key values'.

This is an unstable sort.
*/
fn quick_sort3<T: copy>(compare_func_lt: lteq<T>, compare_func_eq: lteq<T>,
                       arr: [mutable T]) {
    if len::<T>(arr) == 0u { ret; }
    qsort3::<T>(compare_func_lt, compare_func_eq, arr, 0,
                (len::<T>(arr) as int) - 1);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

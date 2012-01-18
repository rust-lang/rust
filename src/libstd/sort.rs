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

#[cfg(test)]
mod test_qsort3 {
    fn check_sort(v1: [mutable int], v2: [mutable int]) {
        let len = vec::len::<int>(v1);
        fn lt(&&a: int, &&b: int) -> bool { ret a < b; }
        fn equal(&&a: int, &&b: int) -> bool { ret a == b; }
        let f1 = lt;
        let f2 = equal;
        quick_sort3::<int>(f1, f2, v1);
        let i = 0u;
        while i < len {
            log(debug, v2[i]);
            assert (v2[i] == v1[i]);
            i += 1u;
        }
    }

    #[test]
    fn test() {
        {
            let v1 = [mutable 3, 7, 4, 5, 2, 9, 5, 8];
            let v2 = [mutable 2, 3, 4, 5, 5, 7, 8, 9];
            check_sort(v1, v2);
        }
        {
            let v1 = [mutable 1, 1, 1];
            let v2 = [mutable 1, 1, 1];
            check_sort(v1, v2);
        }
        {
            let v1: [mutable int] = [mutable];
            let v2: [mutable int] = [mutable];
            check_sort(v1, v2);
        }
        { let v1 = [mutable 9]; let v2 = [mutable 9]; check_sort(v1, v2); }
        {
            let v1 = [mutable 9, 3, 3, 3, 9];
            let v2 = [mutable 3, 3, 3, 9, 9];
            check_sort(v1, v2);
        }
    }
}

#[cfg(test)]
mod test_qsort {
    fn check_sort(v1: [mutable int], v2: [mutable int]) {
        let len = vec::len::<int>(v1);
        fn ltequal(&&a: int, &&b: int) -> bool { ret a <= b; }
        let f = ltequal;
        quick_sort::<int>(f, v1);
        let i = 0u;
        while i < len {
            log(debug, v2[i]);
            assert (v2[i] == v1[i]);
            i += 1u;
        }
    }

    #[test]
    fn test() {
        {
            let v1 = [mutable 3, 7, 4, 5, 2, 9, 5, 8];
            let v2 = [mutable 2, 3, 4, 5, 5, 7, 8, 9];
            check_sort(v1, v2);
        }
        {
            let v1 = [mutable 1, 1, 1];
            let v2 = [mutable 1, 1, 1];
            check_sort(v1, v2);
        }
        {
            let v1: [mutable int] = [mutable];
            let v2: [mutable int] = [mutable];
            check_sort(v1, v2);
        }
        { let v1 = [mutable 9]; let v2 = [mutable 9]; check_sort(v1, v2); }
        {
            let v1 = [mutable 9, 3, 3, 3, 9];
            let v2 = [mutable 3, 3, 3, 9, 9];
            check_sort(v1, v2);
        }
    }

    // Regression test for #750
    #[test]
    fn test_simple() {
        let names = [mutable 2, 1, 3];

        let expected = [1, 2, 3];

        fn lteq(&&a: int, &&b: int) -> bool { int::le(a, b) }
        sort::quick_sort(lteq, names);

        let immut_names = vec::from_mut(names);

        // Silly, but what else can we do?
        check (vec::same_length(expected, immut_names));
        let pairs = vec::zip(expected, immut_names);
        for (a, b) in pairs { #debug("%d %d", a, b); assert (a == b); }
    }
}

#[cfg(test)]
mod tests {

    fn check_sort(v1: [int], v2: [int]) {
        let len = vec::len::<int>(v1);
        fn lteq(&&a: int, &&b: int) -> bool { ret a <= b; }
        let f = lteq;
        let v3 = merge_sort::<int>(f, v1);
        let i = 0u;
        while i < len {
            log(debug, v3[i]);
            assert (v3[i] == v2[i]);
            i += 1u;
        }
    }

    #[test]
    fn test() {
        {
            let v1 = [3, 7, 4, 5, 2, 9, 5, 8];
            let v2 = [2, 3, 4, 5, 5, 7, 8, 9];
            check_sort(v1, v2);
        }
        { let v1 = [1, 1, 1]; let v2 = [1, 1, 1]; check_sort(v1, v2); }
        { let v1: [int] = []; let v2: [int] = []; check_sort(v1, v2); }
        { let v1 = [9]; let v2 = [9]; check_sort(v1, v2); }
        {
            let v1 = [9, 3, 3, 3, 9];
            let v2 = [3, 3, 3, 9, 9];
            check_sort(v1, v2);
        }
    }

    #[test]
    fn test_merge_sort_mutable() {
        fn lteq(&&a: int, &&b: int) -> bool { ret a <= b; }
        let v1 = [mutable 3, 2, 1];
        let v2 = merge_sort(lteq, v1);
        assert v2 == [1, 2, 3];
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

//! Sorting methods
import vec::{len, push};
import core::cmp::{Eq, Ord};

export le;
export merge_sort;
export quick_sort;
export quick_sort3;

type le<T> = pure fn(v1: &T, v2: &T) -> bool;

/**
 * Merge sort. Returns a new vector containing the sorted list.
 *
 * Has worst case O(n log n) performance, best case O(n), but
 * is not space efficient. This is a stable sort.
 */
fn merge_sort<T: copy>(le: le<T>, v: ~[const T]) -> ~[T] {
    type slice = (uint, uint);

    return merge_sort_(le, v, (0u, len(v)));

    fn merge_sort_<T: copy>(le: le<T>, v: ~[const T], slice: slice)
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

    fn merge<T: copy>(le: le<T>, a: ~[T], b: ~[T]) -> ~[T] {
        let mut rs = ~[];
        vec::reserve(rs, len(a) + len(b));
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

fn part<T: copy>(compare_func: le<T>, arr: ~[mut T], left: uint,
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

fn qsort<T: copy>(compare_func: le<T>, arr: ~[mut T], left: uint,
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
fn quick_sort<T: copy>(compare_func: le<T>, arr: ~[mut T]) {
    if len::<T>(arr) == 0u { return; }
    qsort::<T>(compare_func, arr, 0u, len::<T>(arr) - 1u);
}

fn qsort3<T: copy Ord Eq>(arr: ~[mut T], left: int, right: int) {
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
fn quick_sort3<T: copy Ord Eq>(arr: ~[mut T]) {
    if arr.len() <= 1 { return; }
    qsort3(arr, 0, (arr.len() - 1) as int);
}

#[cfg(test)]
mod test_qsort3 {
    fn check_sort(v1: ~[mut int], v2: ~[mut int]) {
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
    fn check_sort(v1: ~[mut int], v2: ~[mut int]) {
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
            let (a, b) = p;
            debug!("%d %d", a, b);
            assert (a == b);
        }
    }
}

#[cfg(test)]
mod tests {

    fn check_sort(v1: ~[int], v2: ~[int]) {
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
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

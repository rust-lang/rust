
import vec::len;
import vec::slice;
import ilen = ivec::len;
import islice = ivec::slice;
export ivector;
export lteq;
export merge_sort;
export quick_sort;
export quick_sort3;

type lteq[T] = fn(&T, &T) -> bool ;

fn merge_sort[T](lteq[T] le, vec[T] v) -> vec[T] {
    fn merge[T](lteq[T] le, vec[T] a, vec[T] b) -> vec[T] {
        let vec[T] rs = [];
        let uint a_len = len[T](a);
        let uint a_ix = 0u;
        let uint b_len = len[T](b);
        let uint b_ix = 0u;
        while (a_ix < a_len && b_ix < b_len) {
            if (le(a.(a_ix), b.(b_ix))) {
                rs += [a.(a_ix)];
                a_ix += 1u;
            } else { rs += [b.(b_ix)]; b_ix += 1u; }
        }
        rs += slice[T](a, a_ix, a_len);
        rs += slice[T](b, b_ix, b_len);
        ret rs;
    }
    let uint v_len = len[T](v);
    if (v_len <= 1u) { ret v; }
    let uint mid = v_len / 2u;
    let vec[T] a = slice[T](v, 0u, mid);
    let vec[T] b = slice[T](v, mid, v_len);
    ret merge[T](le, merge_sort[T](le, a), merge_sort[T](le, b));
}

fn swap[T](vec[mutable T] arr, uint x, uint y) {
    auto a = arr.(x);
    arr.(x) = arr.(y);
    arr.(y) = a;
}

fn part[T](lteq[T] compare_func, vec[mutable T] arr, uint left, uint right,
           uint pivot) -> uint {
    auto pivot_value = arr.(pivot);
    swap[T](arr, pivot, right);
    let uint storage_index = left;
    let uint i = left;
    while (i < right) {
        if (compare_func({ arr.(i) }, pivot_value)) {
            swap[T](arr, i, storage_index);
            storage_index += 1u;
        }
        i += 1u;
    }
    swap[T](arr, storage_index, right);
    ret storage_index;
}

fn qsort[T](lteq[T] compare_func, vec[mutable T] arr, uint left, uint right) {
    if (right > left) {
        auto pivot = (left + right) / 2u;
        auto new_pivot = part[T](compare_func, arr, left, right, pivot);
        if (new_pivot != 0u) {
            // Need to do this check before recursing due to overflow
            qsort[T](compare_func, arr, left, new_pivot - 1u);
        }
        qsort[T](compare_func, arr, new_pivot + 1u, right);
    }
}

fn quick_sort[T](lteq[T] compare_func, vec[mutable T] arr) {
    if (len[T](arr) == 0u) { ret; }
    qsort[T](compare_func, arr, 0u, len[T](arr) - 1u);
}


// Based on algorithm presented by Sedgewick and Bentley here:
// http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
// According to these slides this is the algorithm of choice for
// 'randomly ordered keys, abstract compare' & 'small number of key values'
fn qsort3[T](lteq[T] compare_func_lt, lteq[T] compare_func_eq,
             vec[mutable T] arr, int left, int right) {
    if (right <= left) { ret; }
    let T v = arr.(right);
    let int i = left - 1;
    let int j = right;
    let int p = i;
    let int q = j;
    while (true) {
        i += 1;
        while (compare_func_lt({ arr.(i) }, v)) { i += 1; }
        j -= 1;
        while (compare_func_lt(v, { arr.(j) })) {
            if (j == left) { break; }
            j -= 1;
        }
        if (i >= j) { break; }
        swap[T](arr, i as uint, j as uint);
        if (compare_func_eq({ arr.(i) }, v)) {
            p += 1;
            swap[T](arr, p as uint, i as uint);
        }
        if (compare_func_eq(v, { arr.(j) })) {
            q -= 1;
            swap[T](arr, j as uint, q as uint);
        }
    }
    swap[T](arr, i as uint, right as uint);
    j = i - 1;
    i += 1;
    let int k = left;
    while (k < p) {
        swap[T](arr, k as uint, j as uint);
        k += 1;
        j -= 1;
        if (k == vec::len[T](arr) as int) { break; }
    }
    k = right - 1;
    while (k > q) {
        swap[T](arr, i as uint, k as uint);
        k -= 1;
        i += 1;
        if (k == 0) { break; }
    }
    qsort3[T](compare_func_lt, compare_func_eq, arr, left, j);
    qsort3[T](compare_func_lt, compare_func_eq, arr, i, right);
}

fn quick_sort3[T](lteq[T] compare_func_lt, lteq[T] compare_func_eq,
                  vec[mutable T] arr) {
    if (vec::len[T](arr) == 0u) { ret; }
    qsort3[T](compare_func_lt, compare_func_eq, arr, 0,
              (vec::len[T](arr) as int) - 1);
}

mod ivector {
    export merge_sort;
    export quick_sort;
    export quick_sort3;

    type lteq[T] = fn(&T, &T) -> bool;

    fn merge_sort[T](lteq[T] le, &T[] v) -> T[] {
        fn merge[T](lteq[T] le, &T[] a, &T[] b) -> T[] {
            let T[] rs = ~[];
            let uint a_len = ilen[T](a);
            let uint a_ix = 0u;
            let uint b_len = ilen[T](b);
            let uint b_ix = 0u;
            while (a_ix < a_len && b_ix < b_len) {
                if (le(a.(a_ix), b.(b_ix))) {
                    rs += ~[a.(a_ix)];
                    a_ix += 1u;
                } else { rs += ~[b.(b_ix)]; b_ix += 1u; }
            }
            rs += islice[T](a, a_ix, a_len);
            rs += islice[T](b, b_ix, b_len);
            ret rs;
        }
        let uint v_len = ilen[T](v);
        if (v_len <= 1u) { ret v; }
        let uint mid = v_len / 2u;
        let T[] a = islice[T](v, 0u, mid);
        let T[] b = islice[T](v, mid, v_len);
        ret merge[T](le, merge_sort[T](le, a), merge_sort[T](le, b));
    }

    fn swap[T](&T[mutable] arr, uint x, uint y) {
        auto a = arr.(x);
        arr.(x) = arr.(y);
        arr.(y) = a;
    }

    fn part[T](lteq[T] compare_func, &T[mutable] arr, uint left, uint right,
               uint pivot) -> uint {
        auto pivot_value = arr.(pivot);
        swap[T](arr, pivot, right);
        let uint storage_index = left;
        let uint i = left;
        while (i < right) {
            if (compare_func({ arr.(i) }, pivot_value)) {
                swap[T](arr, i, storage_index);
                storage_index += 1u;
            }
            i += 1u;
        }
        swap[T](arr, storage_index, right);
        ret storage_index;
    }

    fn qsort[T](lteq[T] compare_func, &T[mutable] arr, uint left,
                uint right) {
        if (right > left) {
            auto pivot = (left + right) / 2u;
            auto new_pivot = part[T](compare_func, arr, left, right, pivot);
            if (new_pivot != 0u) {
                // Need to do this check before recursing due to overflow
                qsort[T](compare_func, arr, left, new_pivot - 1u);
            }
            qsort[T](compare_func, arr, new_pivot + 1u, right);
        }
    }

    fn quick_sort[T](lteq[T] compare_func, &T[mutable] arr) {
        if (ilen[T](arr) == 0u) { ret; }
        qsort[T](compare_func, arr, 0u, ilen[T](arr) - 1u);
    }


    // Based on algorithm presented by Sedgewick and Bentley here:
    // http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
    // According to these slides this is the algorithm of choice for
    // 'randomly ordered keys, abstract compare' & 'small number of key
    // values'
    fn qsort3[T](lteq[T] compare_func_lt, lteq[T] compare_func_eq,
                 &T[mutable] arr, int left, int right) {
        if (right <= left) { ret; }
        let T v = arr.(right);
        let int i = left - 1;
        let int j = right;
        let int p = i;
        let int q = j;
        while (true) {
            i += 1;
            while (compare_func_lt({ arr.(i) }, v)) { i += 1; }
            j -= 1;
            while (compare_func_lt(v, { arr.(j) })) {
                if (j == left) { break; }
                j -= 1;
            }
            if (i >= j) { break; }
            swap[T](arr, i as uint, j as uint);
            if (compare_func_eq({ arr.(i) }, v)) {
                p += 1;
                swap[T](arr, p as uint, i as uint);
            }
            if (compare_func_eq(v, { arr.(j) })) {
                q -= 1;
                swap[T](arr, j as uint, q as uint);
            }
        }
        swap[T](arr, i as uint, right as uint);
        j = i - 1;
        i += 1;
        let int k = left;
        while (k < p) {
            swap[T](arr, k as uint, j as uint);
            k += 1;
            j -= 1;
            if (k == ilen[T](arr) as int) { break; }
        }
        k = right - 1;
        while (k > q) {
            swap[T](arr, i as uint, k as uint);
            k -= 1;
            i += 1;
            if (k == 0) { break; }
        }
        qsort3[T](compare_func_lt, compare_func_eq, arr, left, j);
        qsort3[T](compare_func_lt, compare_func_eq, arr, i, right);
    }

    fn quick_sort3[T](lteq[T] compare_func_lt, lteq[T] compare_func_eq,
                      &T[mutable] arr) {
        if (ilen[T](arr) == 0u) { ret; }
        qsort3[T](compare_func_lt, compare_func_eq, arr, 0,
                  (ilen[T](arr) as int) - 1);
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

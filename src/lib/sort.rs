import vec::len;
import vec::slice;

export lteq;
export merge_sort;
export quick_sort;

type lteq[T] = fn(&T a, &T b) -> bool;

fn merge_sort[T](lteq[T] le, vec[T] v) -> vec[T] {

    fn merge[T](lteq[T] le, vec[T] a, vec[T] b) -> vec[T] {
        let vec[T] res = [];
        let uint a_len = len[T](a);
        let uint a_ix = 0u;
        let uint b_len = len[T](b);
        let uint b_ix = 0u;
        while (a_ix < a_len && b_ix < b_len) {
            if (le(a.(a_ix), b.(b_ix))) {
                res += [a.(a_ix)];
                a_ix += 1u;
            } else {
                res += [b.(b_ix)];
                b_ix += 1u;
            }
        }
        res += slice[T](a, a_ix, a_len);
        res += slice[T](b, b_ix, b_len);
        ret res;
    }

    let uint v_len = len[T](v);

    if (v_len <= 1u) {
        ret v;
    }

    let uint mid = v_len / 2u;
    let vec[T] a = slice[T](v, 0u, mid);
    let vec[T] b = slice[T](v, mid, v_len);
    ret merge[T](le,
                 merge_sort[T](le, a),
                 merge_sort[T](le, b));
}

fn swap[T](vec[mutable T] arr, uint x, uint y) {
    auto a = arr.(x);
    arr.(x) = arr.(y);
    arr.(y) = a;
}

fn part[T](lteq[T] compare_func, vec[mutable T] arr, uint left,
        uint right, uint pivot) -> uint {

    fn compare[T](lteq[T] compare_func, vec[mutable T]arr,
           uint arr_idx, &T arr_value) -> bool {

        ret compare_func(arr.(arr_idx),arr_value);
    }

    auto pivot_value = arr.(pivot);
    swap[T](arr, pivot, right);
    let uint storage_index = left;
    let uint i = left;
    while (i<right) {
        if (compare[T](compare_func, arr, i, pivot_value)) {
           swap[T](arr, i, storage_index);
           storage_index += 1u;
        }
        i += 1u;
    }
    swap[T](arr, storage_index, right);
    ret storage_index;
}

fn qsort[T](lteq[T] compare_func, vec[mutable T] arr, uint left,
        uint right) {

    if (right > left) {
        auto pivot = (left+right)/2u;
        auto new_pivot = part[T](compare_func, arr, left, right, pivot);
        if (new_pivot == 0u) {
             ret;
        }
        qsort[T](compare_func, arr, left, new_pivot - 1u);
        qsort[T](compare_func, arr, new_pivot + 1u, right);
    }
}

fn quick_sort[T](lteq[T] compare_func, vec[mutable T] arr) {

    if (len[T](arr) == 0u) {
        ret;
    }
    qsort[T](compare_func, arr, 0u, (len[T](arr)) - 1u);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

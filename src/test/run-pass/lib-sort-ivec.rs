// xfail-stage0

use std;

fn check_sort(&int[] v1, &int[] v2) {
    auto len = std::ivec::len[int](v1);
    fn lteq(&int a, &int b) -> bool { ret a <= b; }
    auto f = lteq;
    auto v3 = std::sort::ivector::merge_sort[int](f, v1);
    auto i = 0u;
    while (i < len) { log v3.(i); assert (v3.(i) == v2.(i)); i += 1u; }
}

fn main() {
    {
        auto v1 = ~[3, 7, 4, 5, 2, 9, 5, 8];
        auto v2 = ~[2, 3, 4, 5, 5, 7, 8, 9];
        check_sort(v1, v2);
    }
    { auto v1 = ~[1, 1, 1]; auto v2 = ~[1, 1, 1]; check_sort(v1, v2); }
    { let int[] v1 = ~[]; let int[] v2 = ~[]; check_sort(v1, v2); }
    { auto v1 = ~[9]; auto v2 = ~[9]; check_sort(v1, v2); }
    {
        auto v1 = ~[9, 3, 3, 3, 9];
        auto v2 = ~[3, 3, 3, 9, 9];
        check_sort(v1, v2);
    }
}

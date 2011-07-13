
use std;

fn check_sort(vec[int] v1, vec[int] v2) {
    auto len = std::vec::len[int](v1);
    fn lteq(&int a, &int b) -> bool { ret a <= b; }
    auto f = lteq;
    auto v3 = std::sort::merge_sort[int](f, v1);
    auto i = 0u;
    while (i < len) { log v3.(i); assert (v3.(i) == v2.(i)); i += 1u; }
}

#[test]
fn test() {
    {
        auto v1 = [3, 7, 4, 5, 2, 9, 5, 8];
        auto v2 = [2, 3, 4, 5, 5, 7, 8, 9];
        check_sort(v1, v2);
    }
    { auto v1 = [1, 1, 1]; auto v2 = [1, 1, 1]; check_sort(v1, v2); }
    { let vec[int] v1 = []; let vec[int] v2 = []; check_sort(v1, v2); }
    { auto v1 = [9]; auto v2 = [9]; check_sort(v1, v2); }
    {
        auto v1 = [9, 3, 3, 3, 9];
        auto v2 = [3, 3, 3, 9, 9];
        check_sort(v1, v2);
    }
}

use std;

import std::sort;
import std::ivec;
import std::int;

fn check_sort(vec[mutable int] v1, vec[mutable int] v2) {
    auto len = std::vec::len[int](v1);
    fn ltequal(&int a, &int b) -> bool { ret a <= b; }
    auto f = ltequal;
    std::sort::quick_sort[int](f, v1);
    auto i = 0u;
    while (i < len) { log v2.(i); assert (v2.(i) == v1.(i)); i += 1u; }
}

#[test]
fn test() {
    {
        auto v1 = [mutable 3, 7, 4, 5, 2, 9, 5, 8];
        auto v2 = [mutable 2, 3, 4, 5, 5, 7, 8, 9];
        check_sort(v1, v2);
    }
    {
        auto v1 = [mutable 1, 1, 1];
        auto v2 = [mutable 1, 1, 1];
        check_sort(v1, v2);
    }
    {
        let vec[mutable int] v1 = [mutable ];
        let vec[mutable int] v2 = [mutable ];
        check_sort(v1, v2);
    }
    { auto v1 = [mutable 9]; auto v2 = [mutable 9]; check_sort(v1, v2); }
    {
        auto v1 = [mutable 9, 3, 3, 3, 9];
        auto v2 = [mutable 3, 3, 3, 9, 9];
        check_sort(v1, v2);
    }
}

// Regression test for #705
#[test]
fn test_simple() {
    auto names = ~[mutable 2, 1, 3];

    auto expected = ~[1, 2, 3];

    fn lteq(&int a, &int b) -> bool { int::le(a, b) }
    sort::ivector::quick_sort(lteq, names);

    auto pairs = ivec::zip(expected, ivec::from_mut(names));
    for (rec(int _0, int _1) p in pairs) {
        log #fmt("%d %d", p._0, p._1);
        assert p._0 == p._1;
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

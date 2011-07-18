use std;
import std::ivec;
import std::int;
import std::sort;

fn test_qsort() {
    auto names = ~[mutable 2, 1, 3];

    auto expected = ~[1, 2, 3];

    fn lteq(&int a, &int b) -> bool { int::le(a, b) }
    sort::ivector::quick_sort(lteq, names);

    auto pairs = ivec::zip(expected, ivec::from_mut(names));
    for (tup(int, int) p in pairs) {
        log_err #fmt("%d %d", p._0, p._1);
        assert p._0 == p._1;
    }
}

fn main() {
    test_qsort();
}

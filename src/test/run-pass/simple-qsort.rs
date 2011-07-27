// xfail-stage0

use std;
import std::ivec;
import std::int;
import std::sort;

fn test_qsort() {
    let names = ~[mutable 2, 1, 3];

    let expected = ~[1, 2, 3];

    fn lteq(a: &int, b: &int) -> bool { int::le(a, b) }
    sort::ivector::quick_sort(lteq, names);

    let pairs = ivec::zip(expected, ivec::from_mut(names));
    for p: {_0: int, _1: int}  in pairs {
        log_err #fmt("%d %d", p._0, p._1);
        assert (p._0 == p._1);
    }
}

fn main() { test_qsort(); }
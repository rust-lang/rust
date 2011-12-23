import core::*;

use std;

fn check_sort(v1: [int], v2: [int]) {
    let len = vec::len::<int>(v1);
    fn lteq(&&a: int, &&b: int) -> bool { ret a <= b; }
    let f = lteq;
    let v3 = std::sort::merge_sort::<int>(f, v1);
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
    let v2 = std::sort::merge_sort(lteq, v1);
    assert v2 == [1, 2, 3];
}
use std::fmt::Debug;
use std::ops::{Range, RangeBounds};
use std::slice;

fn check_is_partial_sorted<T: Ord + Clone + Debug, R: RangeBounds<usize>>(v: &mut [T], range: R) {
    let Range { start, end } = slice::range(range, ..v.len());
    v.partial_sort_unstable(start..end);

    let max_before = v[..start].iter().max().into_iter();
    let sorted_range = v[start..end].into_iter();
    let min_after = v[end..].iter().min().into_iter();
    let seq = max_before.chain(sorted_range).chain(min_after);
    assert!(seq.is_sorted());
}

#[test]
fn basic_impl() {
    check_is_partial_sorted::<i32, _>(&mut [], ..);
    check_is_partial_sorted::<(), _>(&mut [], ..);
    check_is_partial_sorted::<(), _>(&mut [()], ..);
    check_is_partial_sorted::<(), _>(&mut [(), ()], ..);
    check_is_partial_sorted::<(), _>(&mut [(), (), ()], ..);
    check_is_partial_sorted::<i32, _>(&mut [], ..);

    check_is_partial_sorted::<i32, _>(&mut [77], ..);
    check_is_partial_sorted::<i32, _>(&mut [2, 3], ..);
    check_is_partial_sorted::<i32, _>(&mut [2, 3, 6], ..);
    check_is_partial_sorted::<i32, _>(&mut [2, 3, 99, 6], ..);
    check_is_partial_sorted::<i32, _>(&mut [2, 7709, 400, 90932], ..);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], ..);

    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 0..0);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 0..1);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 0..5);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 0..7);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 7..7);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 6..7);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 5..7);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 5..5);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 4..5);
    check_is_partial_sorted::<i32, _>(&mut [15, -1, 3, -1, -3, -1, 7], 4..6);
}

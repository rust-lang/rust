use std::fmt::Debug;
use std::ops::{Range, RangeBounds};
use std::slice;

use super::patterns;

fn check_is_partial_sorted<T: Ord + Clone + Debug, R: RangeBounds<usize>>(v: &mut [T], range: R) {
    let Range { start, end } = slice::range(range, ..v.len());
    v.partial_sort_unstable(start..end);

    let max_before = v[..start].iter().max().into_iter();
    let sorted_range = v[start..end].into_iter();
    let min_after = v[end..].iter().min().into_iter();
    let seq = max_before.chain(sorted_range).chain(min_after);
    assert!(seq.is_sorted());
}

fn check_is_partial_sorted_ranges<T: Ord + Clone + Debug>(v: &[T]) {
    let len = v.len();

    check_is_partial_sorted::<T, _>(&mut v.to_vec(), ..);
    check_is_partial_sorted::<T, _>(&mut v.to_vec(), 0..0);
    check_is_partial_sorted::<T, _>(&mut v.to_vec(), len..len);

    if len > 0 {
        check_is_partial_sorted::<T, _>(&mut v.to_vec(), len - 1..len - 1);
        check_is_partial_sorted::<T, _>(&mut v.to_vec(), 0..1);
        check_is_partial_sorted::<T, _>(&mut v.to_vec(), len - 1..len);

        for mid in 1..len {
            check_is_partial_sorted::<T, _>(&mut v.to_vec(), 0..mid);
            check_is_partial_sorted::<T, _>(&mut v.to_vec(), mid..len);
            check_is_partial_sorted::<T, _>(&mut v.to_vec(), mid..mid);
            check_is_partial_sorted::<T, _>(&mut v.to_vec(), mid - 1..mid + 1);
            check_is_partial_sorted::<T, _>(&mut v.to_vec(), mid - 1..mid);
            check_is_partial_sorted::<T, _>(&mut v.to_vec(), mid..mid + 1);
        }

        let quarters = [0, len / 4, len / 2, (3 * len) / 4, len];
        for &start in &quarters {
            for &end in &quarters {
                if start < end {
                    check_is_partial_sorted::<T, _>(&mut v.to_vec(), start..end);
                }
            }
        }
    }
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

#[test]
fn random_patterns() {
    check_is_partial_sorted_ranges(&patterns::random(10));
    check_is_partial_sorted_ranges(&patterns::random(50));
    check_is_partial_sorted_ranges(&patterns::random(100));
    check_is_partial_sorted_ranges(&patterns::random(1000));
}

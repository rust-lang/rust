use core::iter::*;

#[test]
fn test_filter_map() {
    let it = (0..).step_by(1).take(10).filter_map(|x| if x % 2 == 0 { Some(x * x) } else { None });
    assert_eq!(it.collect::<Vec<usize>>(), [0 * 0, 2 * 2, 4 * 4, 6 * 6, 8 * 8]);
}

#[test]
fn test_filter_map_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [0 * 0, 2 * 2, 4 * 4, 6 * 6, 8 * 8];
    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x * x) } else { None });
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x * x) } else { None });
    let i = it.rfold(ys.len(), |i, x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_filter_map_try_folds() {
    let mp = &|x| if 0 <= x && x < 10 { Some(x * 2) } else { None };
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((-9..20).filter_map(mp).try_fold(7, f), (0..10).map(|x| 2 * x).try_fold(7, f));
    assert_eq!((-9..20).filter_map(mp).try_rfold(7, f), (0..10).map(|x| 2 * x).try_rfold(7, f));

    let mut iter = (0..40).filter_map(|x| if x % 2 == 1 { None } else { Some(x * 2 + 10) });
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(38));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(78));
}

#[test]
fn test_double_ended_filter_map() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter_map(|&x| if x & 1 == 0 { Some(x * 2) } else { None });
    assert_eq!(it.next_back().unwrap(), 12);
    assert_eq!(it.next_back().unwrap(), 8);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.next_back(), None);
}

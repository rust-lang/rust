use core::iter::*;

#[test]
fn test_find_map() {
    let xs: &[isize] = &[];
    assert_eq!(xs.iter().find_map(half_if_even), None);
    let xs: &[isize] = &[3, 5];
    assert_eq!(xs.iter().find_map(half_if_even), None);
    let xs: &[isize] = &[4, 5];
    assert_eq!(xs.iter().find_map(half_if_even), Some(2));
    let xs: &[isize] = &[3, 6];
    assert_eq!(xs.iter().find_map(half_if_even), Some(3));

    let xs: &[isize] = &[1, 2, 3, 4, 5, 6, 7];
    let mut iter = xs.iter();
    assert_eq!(iter.find_map(half_if_even), Some(1));
    assert_eq!(iter.find_map(half_if_even), Some(2));
    assert_eq!(iter.find_map(half_if_even), Some(3));
    assert_eq!(iter.next(), Some(&7));

    fn half_if_even(x: &isize) -> Option<isize> {
        if x % 2 == 0 { Some(x / 2) } else { None }
    }
}
#[test]
fn test_map_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((0..10).map(|x| x + 3).try_fold(7, f), (3..13).try_fold(7, f));
    assert_eq!((0..10).map(|x| x + 3).try_rfold(7, f), (3..13).try_rfold(7, f));

    let mut iter = (0..40).map(|x| x + 10);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(46));
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

use core::iter::*;

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
fn test_double_ended_map() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().map(|&x| x * -1);
    assert_eq!(it.next(), Some(-1));
    assert_eq!(it.next(), Some(-2));
    assert_eq!(it.next_back(), Some(-6));
    assert_eq!(it.next_back(), Some(-5));
    assert_eq!(it.next(), Some(-3));
    assert_eq!(it.next_back(), Some(-4));
    assert_eq!(it.next(), None);
}

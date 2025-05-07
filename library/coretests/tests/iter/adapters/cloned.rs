use core::iter::*;

#[test]
fn test_cloned() {
    let xs = [2, 4, 6, 8];

    let mut it = xs.iter().cloned();
    assert_eq!(it.len(), 4);
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.len(), 3);
    assert_eq!(it.next(), Some(4));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next_back(), Some(8));
    assert_eq!(it.len(), 1);
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(it.len(), 0);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_cloned_side_effects() {
    let mut count = 0;
    {
        let iter = [1, 2, 3]
            .iter()
            .map(|x| {
                count += 1;
                x
            })
            .cloned()
            .zip(&[1]);
        for _ in iter {}
    }
    assert_eq!(count, 2);
}

#[test]
fn test_cloned_try_folds() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    let f_ref = &|acc, &x| i32::checked_add(2 * acc, x);
    assert_eq!(a.iter().cloned().try_fold(7, f), a.iter().try_fold(7, f_ref));
    assert_eq!(a.iter().cloned().try_rfold(7, f), a.iter().try_rfold(7, f_ref));

    let a = [10, 20, 30, 40, 100, 60, 70, 80, 90];
    let mut iter = a.iter().cloned();
    assert_eq!(iter.try_fold(0_i8, |acc, x| acc.checked_add(x)), None);
    assert_eq!(iter.next(), Some(60));
    let mut iter = a.iter().cloned();
    assert_eq!(iter.try_rfold(0_i8, |acc, x| acc.checked_add(x)), None);
    assert_eq!(iter.next_back(), Some(70));
}

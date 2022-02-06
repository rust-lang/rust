use core::iter::*;

#[test]
fn test_iterator_reject() {
    let mut iter = [0, 1, 2, 3, 4, 5, 6, 7, 8].iter().reject(|&&x| x % 2 == 0);

    assert_eq!(iter.next().unwrap(), &1);
    assert_eq!(iter.next().unwrap(), &3);
    assert_eq!(iter.next().unwrap(), &5);
    assert_eq!(iter.next().unwrap(), &7);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_iterator_reject_count() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(xs.iter().reject(|&&x| x % 2 == 0).count(), 4);
}

#[test]
fn test_iterator_reject_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [1, 3, 5, 7];
    let it = xs.iter().reject(|&&x| x % 2 == 0);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().reject(|&&x| x % 2 == 0);
    let i = it.rfold(ys.len(), |i, &x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_reject_try_folds() {
    fn p(&x: &i32) -> bool {
        x < 0 || x >= 10
    }
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((-10..20).reject(p).try_fold(7, f), (0..10).try_fold(7, f));
    assert_eq!((-10..20).reject(p).try_rfold(7, f), (0..10).try_rfold(7, f));

    let mut iter = (0..40).reject(|&x| x % 2 == 1);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(24));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(30));
}

#[test]
fn test_double_ended_reject() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().reject(|&x| *x & 1 == 0);
    assert_eq!(it.next_back().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &3);
    assert_eq!(it.next().unwrap(), &1);
    assert_eq!(it.next_back(), None);
}

use core::iter::*;
use core::num::NonZero;

#[test]
fn test_iterator_enumerate() {
    let xs = [0, 1, 2, 3, 4, 5];
    let it = xs.iter().enumerate();
    for (i, &x) in it {
        assert_eq!(i, x);
    }
}

#[test]
fn test_iterator_enumerate_nth() {
    let xs = [0, 1, 2, 3, 4, 5];
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(i, x);
    }

    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth(0) {
        assert_eq!(i, x);
    }

    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth(1) {
        assert_eq!(i, x);
    }

    let (i, &x) = xs.iter().enumerate().nth(3).unwrap();
    assert_eq!(i, x);
    assert_eq!(i, 3);
}

#[test]
fn test_iterator_enumerate_nth_back() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth_back(0) {
        assert_eq!(i, x);
    }

    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth_back(1) {
        assert_eq!(i, x);
    }

    let (i, &x) = xs.iter().enumerate().nth_back(3).unwrap();
    assert_eq!(i, x);
    assert_eq!(i, 2);
}

#[test]
fn test_iterator_enumerate_count() {
    let xs = [0, 1, 2, 3, 4, 5];
    assert_eq!(xs.iter().enumerate().count(), 6);
}

#[test]
fn test_iterator_enumerate_advance_by() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().enumerate();
    assert_eq!(it.advance_by(0), Ok(()));
    assert_eq!(it.next(), Some((0, &0)));
    assert_eq!(it.advance_by(1), Ok(()));
    assert_eq!(it.next(), Some((2, &2)));
    assert_eq!(it.advance_by(2), Ok(()));
    assert_eq!(it.next(), Some((5, &5)));
    assert_eq!(it.advance_by(1), Err(NonZero::new(1).unwrap()));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_enumerate_fold() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().enumerate();
    // steal a couple to get an interesting offset
    assert_eq!(it.next(), Some((0, &0)));
    assert_eq!(it.next(), Some((1, &1)));
    let i = it.fold(2, |i, (j, &x)| {
        assert_eq!(i, j);
        assert_eq!(x, xs[j]);
        i + 1
    });
    assert_eq!(i, xs.len());

    let mut it = xs.iter().enumerate();
    assert_eq!(it.next(), Some((0, &0)));
    let i = it.rfold(xs.len() - 1, |i, (j, &x)| {
        assert_eq!(i, j);
        assert_eq!(x, xs[j]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_enumerate_try_folds() {
    let f = &|acc, (i, x)| usize::checked_add(2 * acc, x / (i + 1) + i);
    assert_eq!((9..18).enumerate().try_fold(7, f), (0..9).map(|i| (i, i + 9)).try_fold(7, f));
    assert_eq!((9..18).enumerate().try_rfold(7, f), (0..9).map(|i| (i, i + 9)).try_rfold(7, f));

    let mut iter = (100..200).enumerate();
    let f = &|acc, (i, x)| u8::checked_add(acc, u8::checked_div(x, i as u8 + 1)?);
    assert_eq!(iter.try_fold(0, f), None);
    assert_eq!(iter.next(), Some((7, 107)));
    assert_eq!(iter.try_rfold(0, f), None);
    assert_eq!(iter.next_back(), Some((11, 111)));
}

#[test]
fn test_double_ended_enumerate() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().cloned().enumerate();
    assert_eq!(it.next(), Some((0, 1)));
    assert_eq!(it.next(), Some((1, 2)));
    assert_eq!(it.next_back(), Some((5, 6)));
    assert_eq!(it.next_back(), Some((4, 5)));
    assert_eq!(it.next_back(), Some((3, 4)));
    assert_eq!(it.next_back(), Some((2, 3)));
    assert_eq!(it.next(), None);
}

use core::iter::*;

#[test]
fn test_iterator_take() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0, 1, 2, 3, 5];

    let mut it = xs.iter().take(ys.len());
    let mut i = 0;
    assert_eq!(it.len(), ys.len());
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), ys.len() - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().take(ys.len());
    let mut i = 0;
    assert_eq!(it.len(), ys.len());
    while let Some(&x) = it.next_back() {
        i += 1;
        assert_eq!(x, ys[ys.len() - i]);
        assert_eq!(it.len(), ys.len() - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_take_nth() {
    let xs = [0, 1, 2, 4, 5];
    let mut it = xs.iter();
    {
        let mut take = it.by_ref().take(3);
        let mut i = 0;
        while let Some(&x) = take.nth(0) {
            assert_eq!(x, i);
            i += 1;
        }
    }
    assert_eq!(it.nth(1), Some(&5));
    assert_eq!(it.nth(0), None);

    let xs = [0, 1, 2, 3, 4];
    let mut it = xs.iter().take(7);
    let mut i = 1;
    while let Some(&x) = it.nth(1) {
        assert_eq!(x, i);
        i += 2;
    }
}

#[test]
fn test_iterator_take_nth_back() {
    let xs = [0, 1, 2, 4, 5];
    let mut it = xs.iter();
    {
        let mut take = it.by_ref().take(3);
        let mut i = 0;
        while let Some(&x) = take.nth_back(0) {
            i += 1;
            assert_eq!(x, 3 - i);
        }
    }
    assert_eq!(it.nth_back(0), None);

    let xs = [0, 1, 2, 3, 4];
    let mut it = xs.iter().take(7);
    assert_eq!(it.nth_back(1), Some(&3));
    assert_eq!(it.nth_back(1), Some(&1));
    assert_eq!(it.nth_back(1), None);
}

#[test]
fn test_iterator_take_short() {
    let xs = [0, 1, 2, 3];

    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), xs.len());
    while let Some(&x) = it.next() {
        assert_eq!(x, xs[i]);
        i += 1;
        assert_eq!(it.len(), xs.len() - i);
    }
    assert_eq!(i, xs.len());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), xs.len());
    while let Some(&x) = it.next_back() {
        i += 1;
        assert_eq!(x, xs[xs.len() - i]);
        assert_eq!(it.len(), xs.len() - i);
    }
    assert_eq!(i, xs.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_take_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((10..30).take(10).try_fold(7, f), (10..20).try_fold(7, f));
    assert_eq!((10..30).take(10).try_rfold(7, f), (10..20).try_rfold(7, f));

    let mut iter = (10..30).take(20);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(24));

    let mut iter = (2..20).take(3);
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.try_for_each(Err), Ok(()));

    let mut iter = (2..20).take(3).rev();
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.try_for_each(Err), Ok(()));
}

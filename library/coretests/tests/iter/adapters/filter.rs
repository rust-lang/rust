use core::iter::*;
use std::rc::Rc;

#[test]
fn test_iterator_filter_count() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(xs.iter().filter(|&&x| x % 2 == 0).count(), 5);
}

#[test]
fn test_iterator_filter_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [0, 2, 4, 6, 8];
    let it = xs.iter().filter(|&&x| x % 2 == 0);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().filter(|&&x| x % 2 == 0);
    let i = it.rfold(ys.len(), |i, &x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_filter_try_folds() {
    fn p(&x: &i32) -> bool {
        0 <= x && x < 10
    }
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((-10..20).filter(p).try_fold(7, f), (0..10).try_fold(7, f));
    assert_eq!((-10..20).filter(p).try_rfold(7, f), (0..10).try_rfold(7, f));

    let mut iter = (0..40).filter(|&x| x % 2 == 1);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(25));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(31));
}

#[test]
fn test_double_ended_filter() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter(|&x| *x & 1 == 0);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.next().unwrap(), &2);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_next_chunk_does_not_leak() {
    let drop_witness: [_; 5] = std::array::from_fn(|_| Rc::new(()));

    let v = (0..5).map(|i| drop_witness[i].clone()).collect::<Vec<_>>();
    let _ = v.into_iter().filter(|_| false).next_chunk::<1>();

    for ref w in drop_witness {
        assert_eq!(Rc::strong_count(w), 1);
    }
}

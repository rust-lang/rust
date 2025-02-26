use core::iter::*;

#[test]
fn test_iterator_flat_map() {
    let xs = [0, 3, 6];
    let ys = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let it = xs.iter().flat_map(|&x| (x..).step_by(1).take(3));
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

/// Tests `FlatMap::fold` with items already picked off the front and back,
/// to make sure all parts of the `FlatMap` are folded correctly.
#[test]
fn test_iterator_flat_map_fold() {
    let xs = [0, 3, 6];
    let ys = [1, 2, 3, 4, 5, 6, 7];
    let mut it = xs.iter().flat_map(|&x| x..x + 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().flat_map(|&x| x..x + 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.rfold(ys.len(), |i, x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_flat_map_try_folds() {
    let f = &|acc, x| i32::checked_add(acc * 2 / 3, x);
    let mr = &|x| (5 * x)..(5 * x + 5);
    assert_eq!((0..10).flat_map(mr).try_fold(7, f), (0..50).try_fold(7, f));
    assert_eq!((0..10).flat_map(mr).try_rfold(7, f), (0..50).try_rfold(7, f));
    let mut iter = (0..10).flat_map(mr);
    iter.next();
    iter.next_back(); // have front and back iters in progress
    assert_eq!(iter.try_rfold(7, f), (1..49).try_rfold(7, f));

    let mut iter = (0..10).flat_map(|x| (4 * x)..(4 * x + 4));
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(17));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(35));
}

#[test]
fn test_double_ended_flat_map() {
    let u = [0, 1];
    let v = [5, 6, 7, 8];
    let mut it = u.iter().flat_map(|x| &v[*x..v.len()]);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}

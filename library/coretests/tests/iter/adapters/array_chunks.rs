use core::iter::{self};

use super::*;

#[test]
fn test_iterator_array_chunks_infer() {
    let xs = [1, 1, 2, -2, 6, 0, 3, 1];
    for [a, b, c] in xs.iter().copied().array_chunks() {
        assert_eq!(a + b + c, 4);
    }
}

#[test]
fn test_iterator_array_chunks_clone_and_drop() {
    let count = Cell::new(0);
    let mut it = (0..5).map(|_| CountDrop::new(&count)).array_chunks::<3>();
    assert_eq!(it.by_ref().count(), 1);
    assert_eq!(count.get(), 3);
    let mut it2 = it.clone();
    assert_eq!(count.get(), 3);
    assert_eq!(it.into_remainder().unwrap().len(), 2);
    assert_eq!(count.get(), 5);
    assert!(it2.next().is_none());
    assert_eq!(it2.into_remainder().unwrap().len(), 2);
    assert_eq!(count.get(), 7);
}

#[test]
fn test_iterator_array_chunks_remainder() {
    let mut it = (0..11).array_chunks::<4>();
    assert_eq!(it.next(), Some([0, 1, 2, 3]));
    assert_eq!(it.next(), Some([4, 5, 6, 7]));
    assert_eq!(it.next(), None);
    assert_eq!(it.into_remainder().unwrap().as_slice(), &[8, 9, 10]);
}

#[test]
fn test_iterator_array_chunks_size_hint() {
    let it = (0..6).array_chunks::<1>();
    assert_eq!(it.size_hint(), (6, Some(6)));

    let it = (0..6).array_chunks::<3>();
    assert_eq!(it.size_hint(), (2, Some(2)));

    let it = (0..6).array_chunks::<5>();
    assert_eq!(it.size_hint(), (1, Some(1)));

    let it = (0..6).array_chunks::<7>();
    assert_eq!(it.size_hint(), (0, Some(0)));

    let it = (1..).array_chunks::<2>();
    assert_eq!(it.size_hint(), (usize::MAX / 2, None));

    let it = (1..).filter(|x| x % 2 != 0).array_chunks::<2>();
    assert_eq!(it.size_hint(), (0, None));
}

#[test]
fn test_iterator_array_chunks_count() {
    let it = (0..6).array_chunks::<1>();
    assert_eq!(it.count(), 6);

    let it = (0..6).array_chunks::<3>();
    assert_eq!(it.count(), 2);

    let it = (0..6).array_chunks::<5>();
    assert_eq!(it.count(), 1);

    let it = (0..6).array_chunks::<7>();
    assert_eq!(it.count(), 0);

    let it = (0..6).filter(|x| x % 2 == 0).array_chunks::<2>();
    assert_eq!(it.count(), 1);

    let it = iter::empty::<i32>().array_chunks::<2>();
    assert_eq!(it.count(), 0);

    let it = [(); usize::MAX].iter().array_chunks::<2>();
    assert_eq!(it.count(), usize::MAX / 2);
}

#[test]
fn test_iterator_array_chunks_next_and_next_back() {
    let mut it = (0..11).array_chunks::<3>();
    assert_eq!(it.next(), Some([0, 1, 2]));
    assert_eq!(it.next_back(), Some([6, 7, 8]));
    assert_eq!(it.next(), Some([3, 4, 5]));
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.into_remainder().unwrap().as_slice(), &[9, 10]);
}

#[test]
fn test_iterator_array_chunks_rev_remainder() {
    let mut it = (0..11).array_chunks::<4>();
    {
        let mut it = it.by_ref().rev();
        assert_eq!(it.next(), Some([4, 5, 6, 7]));
        assert_eq!(it.next(), Some([0, 1, 2, 3]));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }
    assert_eq!(it.into_remainder().unwrap().as_slice(), &[8, 9, 10]);
}

#[test]
fn test_iterator_array_chunks_try_fold() {
    let count = Cell::new(0);
    let mut it = (0..10).map(|_| CountDrop::new(&count)).array_chunks::<3>();
    let result: Result<_, ()> = it.by_ref().try_fold(0, |acc, _item| Ok(acc + 1));
    assert_eq!(result, Ok(3));
    assert_eq!(count.get(), 9);
    drop(it);
    assert_eq!(count.get(), 10);

    let count = Cell::new(0);
    let mut it = (0..10).map(|_| CountDrop::new(&count)).array_chunks::<3>();
    let result = it.by_ref().try_fold(0, |acc, _item| if acc < 2 { Ok(acc + 1) } else { Err(acc) });
    assert_eq!(result, Err(2));
    assert_eq!(count.get(), 9);
    drop(it);
    assert_eq!(count.get(), 9);
}

#[test]
fn test_iterator_array_chunks_fold() {
    let result = (1..11).array_chunks::<3>().fold(0, |acc, [a, b, c]| {
        assert_eq!(acc + 1, a);
        assert_eq!(acc + 2, b);
        assert_eq!(acc + 3, c);
        acc + 3
    });
    assert_eq!(result, 9);

    let count = Cell::new(0);
    let result =
        (0..10).map(|_| CountDrop::new(&count)).array_chunks::<3>().fold(0, |acc, _item| acc + 1);
    assert_eq!(result, 3);
    // fold impls may or may not process the remainder
    assert!(count.get() <= 10 && count.get() >= 9);
}

#[test]
fn test_iterator_array_chunks_try_rfold() {
    let count = Cell::new(0);
    let mut it = (0..10).map(|_| CountDrop::new(&count)).array_chunks::<3>();
    let result: Result<_, ()> = it.try_rfold(0, |acc, _item| Ok(acc + 1));
    assert_eq!(result, Ok(3));
    assert_eq!(count.get(), 9);
    drop(it);
    assert_eq!(count.get(), 10);

    let count = Cell::new(0);
    let mut it = (0..10).map(|_| CountDrop::new(&count)).array_chunks::<3>();
    let result = it.try_rfold(0, |acc, _item| if acc < 2 { Ok(acc + 1) } else { Err(acc) });
    assert_eq!(result, Err(2));
    assert_eq!(count.get(), 9);
    drop(it);
    assert_eq!(count.get(), 10);
}

#[test]
fn test_iterator_array_chunks_rfold() {
    let result = (1..11).array_chunks::<3>().rfold(0, |acc, [a, b, c]| {
        assert_eq!(10 - (acc + 1), c);
        assert_eq!(10 - (acc + 2), b);
        assert_eq!(10 - (acc + 3), a);
        acc + 3
    });
    assert_eq!(result, 9);

    let count = Cell::new(0);
    let result =
        (0..10).map(|_| CountDrop::new(&count)).array_chunks::<3>().rfold(0, |acc, _item| acc + 1);
    assert_eq!(result, 3);
    assert_eq!(count.get(), 10);
}

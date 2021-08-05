use core::cell::Cell;
use core::iter::*;

#[derive(Debug, Clone)]
struct DropBomb<'a> {
    dropped: bool,
    counter: &'a Cell<usize>,
}

impl<'a> DropBomb<'a> {
    fn new(counter: &'a Cell<usize>) -> Self {
        Self { dropped: false, counter }
    }
}

impl Drop for DropBomb<'_> {
    fn drop(&mut self) {
        if self.dropped {
            panic!("double dropped!");
        }
        self.dropped = true;
        self.counter.set(self.counter.get() + 1);
    }
}

#[test]
fn test_iterator_array_chunks_clone() {
    let mut iter = (0..=10).array_chunks::<4>();
    let mut iter2 = iter.clone();
    for (x, y) in iter.by_ref().zip(iter2.by_ref()) {
        assert_eq!(x, y);
    }
    assert_eq!(iter.remainder(), &[8, 9, 10]);
    assert_eq!(iter2.remainder(), &[]);
    assert_eq!(iter2.next(), None);
    assert_eq!(iter2.remainder(), &[8, 9, 10]);

    let counter = Cell::new(0);
    let mut iter = once(DropBomb::new(&counter)).cycle().take(11).array_chunks::<3>();
    let mut iter2 = iter.clone();
    for (i, (_x, _y)) in iter.by_ref().zip(iter2.by_ref()).enumerate() {
        assert_eq!(counter.get(), i * 6);
    }
    assert_eq!(counter.get(), 18);
    drop(iter);
    assert_eq!(counter.get(), 21);
    assert_eq!(iter2.remainder().len(), 0);
    assert!(iter2.next().is_none());
    assert_eq!(iter2.remainder().len(), 2);
    drop(iter2);
    assert_eq!(counter.get(), 24);
}

#[test]
fn test_iterator_array_chunks_remainder() {
    let mut iter = (0..=10).array_chunks::<4>();
    assert_eq!(iter.remainder(), &[]);
    assert_eq!(iter.remainder_mut(), &[]);
    assert_eq!(iter.next(), Some([0, 1, 2, 3]));
    assert_eq!(iter.remainder(), &[]);
    assert_eq!(iter.remainder_mut(), &[]);
    assert_eq!(iter.next(), Some([4, 5, 6, 7]));
    assert_eq!(iter.remainder(), &[]);
    assert_eq!(iter.remainder_mut(), &[]);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), &[8, 9, 10]);
    assert_eq!(iter.remainder_mut(), &[8, 9, 10]);
}

#[test]
fn test_iterator_array_chunks_drop() {
    let counter = Cell::new(0);
    let create = |n| (0..n).map(|_| DropBomb::new(&counter)).array_chunks::<3>();

    let iter = create(5);
    assert_eq!(counter.get(), 0);
    drop(iter);
    assert_eq!(counter.get(), 0);

    let mut iter = create(3);
    counter.set(0);
    iter.next();
    assert_eq!(counter.get(), 3);
    assert!(iter.next().is_none());
    assert_eq!(counter.get(), 3);
    assert_eq!(iter.remainder().len(), 0);
    drop(iter);
    assert_eq!(counter.get(), 3);

    let mut iter = create(5);
    counter.set(0);
    iter.next();
    assert_eq!(counter.get(), 3);
    assert!(iter.next().is_none());
    assert_eq!(counter.get(), 3);
    assert_eq!(iter.remainder().len(), 2);
    drop(iter);
    assert_eq!(counter.get(), 5);
}

#[test]
fn test_iterator_array_chunks_try_fold() {
    let mut iter = (0..=10).array_chunks::<3>();
    let result = iter.try_fold(0, |acc, arr| {
        assert_eq!(arr, [acc * 3, (acc * 3) + 1, (acc * 3) + 2]);
        if acc == 2 { Err(acc) } else { Ok(acc + 1) }
    });
    assert_eq!(result, Err(2));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), &[9, 10]);

    let mut iter = (0..10).array_chunks::<2>();
    let result: Result<_, ()> = iter.try_fold(0, |acc, arr| {
        assert_eq!(arr, [acc * 2, (acc * 2) + 1]);
        Ok(acc + 1)
    });
    assert_eq!(result, Ok(5));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), &[]);

    let counter = Cell::new(0);
    let mut iter = (0..=10).map(|_| DropBomb::new(&counter)).array_chunks::<3>();
    let result = iter.try_fold(0, |acc, _arr| {
        assert_eq!(counter.get(), acc * 3);
        if acc == 1 { Err(acc) } else { Ok(acc + 1) }
    });
    assert_eq!(result, Err(1));
    assert_eq!(iter.remainder().len(), 0);
    assert_eq!(counter.get(), 6);
    drop(iter);
    assert_eq!(counter.get(), 6);

    counter.set(0);
    let mut iter = (0..=10).map(|_| DropBomb::new(&counter)).array_chunks::<3>();
    let result: Result<_, ()> = iter.try_fold(0, |acc, _arr| {
        assert_eq!(counter.get(), acc * 3);
        Ok(acc + 1)
    });
    assert_eq!(result, Ok(3));
    assert_eq!(iter.remainder().len(), 2);
    assert_eq!(counter.get(), 9);
    drop(iter);
    assert_eq!(counter.get(), 11);
}

#[test]
fn test_iterator_array_chunks_fold() {
    let result = (0..10).array_chunks::<3>().fold(0, |acc, arr| {
        assert_eq!(arr, [acc * 3, (acc * 3) + 1, (acc * 3) + 2]);
        acc + 1
    });
    assert_eq!(result, 3);

    let counter = Cell::new(0);
    (0..10).map(|_| DropBomb::new(&counter)).array_chunks::<3>().fold(0, |acc, _arr| {
        assert_eq!(counter.get(), acc * 3);
        acc + 1
    });
    assert_eq!(counter.get(), 10);
}

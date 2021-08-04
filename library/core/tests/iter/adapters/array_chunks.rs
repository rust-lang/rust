use core::cell::Cell;
use core::iter::*;

#[derive(Debug)]
struct DropBomb<'a> {
    dropped: bool,
    counter: &'a Cell<usize>,
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
    let create =
        |n| (0..n).map(|_| DropBomb { dropped: false, counter: &counter }).array_chunks::<3>();

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
fn test_iterator_array_rchunks_remainder() {
    let mut iter = (0..=10).array_rchunks::<4>();
    assert_eq!(iter.remainder(), &[]);
    assert_eq!(iter.remainder_mut(), &[]);
    assert_eq!(iter.next(), Some([7, 8, 9, 10]));
    assert_eq!(iter.remainder(), &[]);
    assert_eq!(iter.remainder_mut(), &[]);
    assert_eq!(iter.next(), Some([3, 4, 5, 6]));
    assert_eq!(iter.remainder(), &[]);
    assert_eq!(iter.remainder_mut(), &[]);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.remainder(), &[0, 1, 2]);
    assert_eq!(iter.remainder_mut(), &[0, 1, 2]);
}

#[test]
fn test_iterator_array_rchunks_drop() {
    let counter = Cell::new(0);
    let create =
        |n| (0..n).map(|_| DropBomb { dropped: false, counter: &counter }).array_rchunks::<3>();

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

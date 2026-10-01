use core::iter::*;
use std::num::NonZero;
use crate::iter::Cell;

#[test]
fn test_fuse_nth() {
    let xs = [0, 1, 2];
    let mut it = xs.iter();

    assert_eq!(it.len(), 3);
    assert_eq!(it.nth(2), Some(&2));
    assert_eq!(it.len(), 0);
    assert_eq!(it.nth(2), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_last() {
    let xs = [0, 1, 2];
    let it = xs.iter();

    assert_eq!(it.len(), 3);
    assert_eq!(it.last(), Some(&2));
}

#[test]
fn test_fuse_count() {
    let xs = [0, 1, 2];
    let it = xs.iter();

    assert_eq!(it.len(), 3);
    assert_eq!(it.count(), 3);
    // Can't check len now because count consumes.
}

#[test]
fn test_fuse_fold() {
    let xs = [0, 1, 2];
    let it = xs.iter(); // `FusedIterator`
    let i = it.fuse().fold(0, |i, &x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());

    let it = xs.iter(); // `FusedIterator`
    let i = it.fuse().rfold(xs.len(), |i, &x| {
        assert_eq!(x, xs[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);

    let it = xs.iter().scan((), |_, &x| Some(x)); // `!FusedIterator`
    let i = it.fuse().fold(0, |i, x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_fuse() {
    let mut it = 0..3;
    assert_eq!(it.len(), 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.len(), 1);
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_advance_back_by() {
    // Fused inner iterator: specialized impl
    let mut it = (0..10).fuse();
    assert_eq!(it.advance_back_by(0), Ok(()));
    assert_eq!(it.advance_back_by(3), Ok(())); // drops 9, 8, 7
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(it.advance_back_by(100), Err(NonZero::new(94).unwrap())); // 6 left, 94 short
    assert_eq!(it.next_back(), None);
    assert_eq!(it.advance_back_by(0), Ok(()));
    assert_eq!(it.advance_back_by(1), Err(NonZero::new(1).unwrap()));
}

#[test]
fn test_fuse_advance_back_by_clears_unfused() {
    // Not a FusedIterator: next_back returns None on odd calls, Some on even calls.
    struct Unfused(u32);
    impl Iterator for Unfused {
        type Item = u32;
        fn next(&mut self) -> Option<u32> {
            None
        }
    }
    impl DoubleEndedIterator for Unfused {
        fn next_back(&mut self) -> Option<u32> {
            self.0 += 1;
            if self.0 % 2 == 1 { None } else { Some(self.0) }
        }
    }

    let mut it = Unfused(0).fuse();
    // The first inner call gives None, so the 2 requested steps all remain.
    assert_eq!(it.advance_back_by(2), Err(NonZero::new(2).unwrap()));
    // Unfused would return Some(2) here. Fuse must stay exhausted.
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_fuse_advance_back_by_forwards() {
    // The inner advance_back_by override must be called, not the next_back loop.
    struct Probe<'a>(&'a Cell<usize>);
    impl Iterator for Probe<'_> {
        type Item = ();
        fn next(&mut self) -> Option<()> {
            None
        }
    }
    impl DoubleEndedIterator for Probe<'_> {
        fn next_back(&mut self) -> Option<()> {
            None
        }
        fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
            self.0.set(self.0.get() + 1);
            match NonZero::new(n) {
                Some(n) => Err(n),
                None => Ok(()),
            }
        }
    }
    impl FusedIterator for Probe<'_> {}

    let calls = Cell::new(0);
    let mut it = Probe(&calls).fuse();
    let _ = it.advance_back_by(5);
    assert_eq!(calls.get(), 1);
}

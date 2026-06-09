use core::iter::*;
use core::num::NonZero;

use super::Unfuse;

#[test]
fn test_iterator_skip() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let ys = [13, 15, 16, 17, 19, 20, 30];
    let mut it = xs.iter().skip(5);
    let mut i = 0;
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), xs.len() - 5 - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_skip_doubleended() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let mut it = xs.iter().rev().skip(5);
    assert_eq!(it.next(), Some(&15));
    assert_eq!(it.by_ref().rev().next(), Some(&0));
    assert_eq!(it.next(), Some(&13));
    assert_eq!(it.by_ref().rev().next(), Some(&1));
    assert_eq!(it.next(), Some(&5));
    assert_eq!(it.by_ref().rev().next(), Some(&2));
    assert_eq!(it.next(), Some(&3));
    assert_eq!(it.next(), None);
    let mut it = xs.iter().rev().skip(5).rev();
    assert_eq!(it.next(), Some(&0));
    assert_eq!(it.rev().next(), Some(&15));
    let mut it_base = xs.iter();
    {
        let mut it = it_base.by_ref().skip(5).rev();
        assert_eq!(it.next(), Some(&30));
        assert_eq!(it.next(), Some(&20));
        assert_eq!(it.next(), Some(&19));
        assert_eq!(it.next(), Some(&17));
        assert_eq!(it.next(), Some(&16));
        assert_eq!(it.next(), Some(&15));
        assert_eq!(it.next(), Some(&13));
        assert_eq!(it.next(), None);
    }
    // make sure the skipped parts have not been consumed
    assert_eq!(it_base.next(), Some(&0));
    assert_eq!(it_base.next(), Some(&1));
    assert_eq!(it_base.next(), Some(&2));
    assert_eq!(it_base.next(), Some(&3));
    assert_eq!(it_base.next(), Some(&5));
    assert_eq!(it_base.next(), None);
    let it = xs.iter().skip(5).rev();
    assert_eq!(it.last(), Some(&13));
}

#[test]
fn test_iterator_skip_nth() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];

    let mut it = xs.iter().skip(0);
    assert_eq!(it.nth(0), Some(&0));
    assert_eq!(it.nth(1), Some(&2));

    let mut it = xs.iter().skip(5);
    assert_eq!(it.nth(0), Some(&13));
    assert_eq!(it.nth(1), Some(&16));

    let mut it = xs.iter().skip(12);
    assert_eq!(it.nth(0), None);
}

#[test]
fn test_skip_advance_by() {
    assert_eq!((0..0).skip(10).advance_by(0), Ok(()));
    assert_eq!((0..0).skip(10).advance_by(1), Err(NonZero::new(1).unwrap()));
    assert_eq!(
        (0u128..(usize::MAX as u128) + 1).skip(usize::MAX - 10).advance_by(usize::MAX - 5),
        Err(NonZero::new(usize::MAX - 16).unwrap())
    );
    assert_eq!((0u128..u128::MAX).skip(usize::MAX - 10).advance_by(20), Ok(()));

    assert_eq!((0..2).skip(1).advance_back_by(10), Err(NonZero::new(9).unwrap()));
    assert_eq!((0..0).skip(1).advance_back_by(0), Ok(()));
}

#[test]
fn test_iterator_skip_count() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];

    assert_eq!(xs.iter().skip(0).count(), 12);
    assert_eq!(xs.iter().skip(1).count(), 11);
    assert_eq!(xs.iter().skip(11).count(), 1);
    assert_eq!(xs.iter().skip(12).count(), 0);
    assert_eq!(xs.iter().skip(13).count(), 0);
}

#[test]
fn test_iterator_skip_last() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];

    assert_eq!(xs.iter().skip(0).last(), Some(&30));
    assert_eq!(xs.iter().skip(1).last(), Some(&30));
    assert_eq!(xs.iter().skip(11).last(), Some(&30));
    assert_eq!(xs.iter().skip(12).last(), None);
    assert_eq!(xs.iter().skip(13).last(), None);

    let mut it = xs.iter().skip(5);
    assert_eq!(it.next(), Some(&13));
    assert_eq!(it.last(), Some(&30));
}

#[test]
fn test_iterator_skip_fold() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let ys = [13, 15, 16, 17, 19, 20, 30];

    let it = xs.iter().skip(5);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().skip(5);
    assert_eq!(it.next(), Some(&ys[0])); // process skips before folding
    let i = it.fold(1, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().skip(5);
    let i = it.rfold(ys.len(), |i, &x| {
        let i = i - 1;
        assert_eq!(x, ys[i]);
        i
    });
    assert_eq!(i, 0);

    let mut it = xs.iter().skip(5);
    assert_eq!(it.next(), Some(&ys[0])); // process skips before folding
    let i = it.rfold(ys.len(), |i, &x| {
        let i = i - 1;
        assert_eq!(x, ys[i]);
        i
    });
    assert_eq!(i, 1);
}

#[test]
fn test_skip_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((1..20).skip(9).try_fold(7, f), (10..20).try_fold(7, f));
    assert_eq!((1..20).skip(9).try_rfold(7, f), (10..20).try_rfold(7, f));

    let mut iter = (0..30).skip(10);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(24));
}

#[test]
fn test_skip_nth_back() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().skip(2);
    assert_eq!(it.nth_back(0), Some(&5));
    assert_eq!(it.nth_back(1), Some(&3));
    assert_eq!(it.nth_back(0), Some(&2));
    assert_eq!(it.nth_back(0), None);

    let ys = [2, 3, 4, 5];
    let mut ity = ys.iter();
    let mut it = xs.iter().skip(2);
    assert_eq!(it.nth_back(1), ity.nth_back(1));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));
    assert_eq!(it.nth_back(0), ity.nth_back(0));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));
    assert_eq!(it.nth_back(0), ity.nth_back(0));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));
    assert_eq!(it.nth_back(0), ity.nth_back(0));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));

    let mut it = xs.iter().skip(2);
    assert_eq!(it.nth_back(4), None);
    assert_eq!(it.nth_back(0), None);

    let mut it = xs.iter();
    it.by_ref().skip(2).nth_back(3);
    assert_eq!(it.next_back(), Some(&1));

    let mut it = xs.iter();
    it.by_ref().skip(2).nth_back(10);
    assert_eq!(it.next_back(), Some(&1));
}

#[test]
fn test_skip_non_fused() {
    let non_fused = Unfuse::new(0..10);

    // `Skip` would previously exhaust the iterator in this `next` call and then erroneously try to
    // advance it further. `Unfuse` tests that this doesn't happen by panicking in that scenario.
    let _ = non_fused.skip(20).next();
}

#[test]
fn test_skip_non_fused_nth_overflow() {
    let non_fused = Unfuse::new(0..10);

    // Ensures that calling skip and `nth` where the sum would overflow does not fail for non-fused
    // iterators.
    let _ = non_fused.skip(20).nth(usize::MAX);
}

#[test]
fn test_skip_overflow_wrapping() {
    // Test to ensure even on overflowing on `skip+nth` the correct amount of elements are yielded.
    struct WrappingIterator(usize);

    impl Iterator for WrappingIterator {
        type Item = usize;

        fn next(&mut self) -> core::option::Option<Self::Item> {
            <Self as Iterator>::nth(self, 0)
        }

        fn nth(&mut self, nth: usize) -> core::option::Option<Self::Item> {
            self.0 = self.0.wrapping_add(nth.wrapping_add(1));
            Some(self.0)
        }
    }

    let wrap = WrappingIterator(0);
    assert_eq!(wrap.skip(20).nth(usize::MAX), Some(20));
}

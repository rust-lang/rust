use core::iter::*;
use core::num::NonZero;

use super::*;

#[test]
fn test_chain() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    assert_eq!(Vec::from_iter(chain(xs, ys)), expected);
}

#[test]
fn test_iterator_chain() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    let it = xs.iter().chain(&ys);
    let mut i = 0;
    for &x in it {
        assert_eq!(x, expected[i]);
        i += 1;
    }
    assert_eq!(i, expected.len());

    let ys = (30..).step_by(10).take(4);
    let it = xs.iter().cloned().chain(ys);
    let mut i = 0;
    for x in it {
        assert_eq!(x, expected[i]);
        i += 1;
    }
    assert_eq!(i, expected.len());
}

#[test]
fn test_iterator_chain_advance_by() {
    fn test_chain(xs: &[i32], ys: &[i32]) {
        let len = xs.len() + ys.len();

        for i in 0..xs.len() {
            let mut iter = Unfuse::new(xs).chain(Unfuse::new(ys));
            assert_eq!(iter.advance_by(i), Ok(()));
            assert_eq!(iter.next(), Some(&xs[i]));
            assert_eq!(iter.advance_by(100), Err(NonZero::new(100 - (len - i - 1)).unwrap()));
            assert_eq!(iter.advance_by(0), Ok(()));
        }

        for i in 0..ys.len() {
            let mut iter = Unfuse::new(xs).chain(Unfuse::new(ys));
            assert_eq!(iter.advance_by(xs.len() + i), Ok(()));
            assert_eq!(iter.next(), Some(&ys[i]));
            assert_eq!(iter.advance_by(100), Err(NonZero::new(100 - (ys.len() - i - 1)).unwrap()));
            assert_eq!(iter.advance_by(0), Ok(()));
        }

        let mut iter = xs.iter().chain(ys);
        assert_eq!(iter.advance_by(len), Ok(()));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.advance_by(0), Ok(()));

        let mut iter = xs.iter().chain(ys);
        assert_eq!(iter.advance_by(len + 1), Err(NonZero::new(1).unwrap()));
        assert_eq!(iter.advance_by(0), Ok(()));
    }

    test_chain(&[], &[]);
    test_chain(&[], &[0, 1, 2, 3, 4, 5]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[30, 40, 50, 60]);
}

#[test]
fn test_iterator_chain_advance_back_by() {
    fn test_chain(xs: &[i32], ys: &[i32]) {
        let len = xs.len() + ys.len();

        for i in 0..ys.len() {
            let mut iter = Unfuse::new(xs).chain(Unfuse::new(ys));
            assert_eq!(iter.advance_back_by(i), Ok(()));
            assert_eq!(iter.next_back(), Some(&ys[ys.len() - i - 1]));
            assert_eq!(iter.advance_back_by(100), Err(NonZero::new(100 - (len - i - 1)).unwrap()));
            assert_eq!(iter.advance_back_by(0), Ok(()));
        }

        for i in 0..xs.len() {
            let mut iter = Unfuse::new(xs).chain(Unfuse::new(ys));
            assert_eq!(iter.advance_back_by(ys.len() + i), Ok(()));
            assert_eq!(iter.next_back(), Some(&xs[xs.len() - i - 1]));
            assert_eq!(
                iter.advance_back_by(100),
                Err(NonZero::new(100 - (xs.len() - i - 1)).unwrap())
            );
            assert_eq!(iter.advance_back_by(0), Ok(()));
        }

        let mut iter = xs.iter().chain(ys);
        assert_eq!(iter.advance_back_by(len), Ok(()));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.advance_back_by(0), Ok(()));

        let mut iter = xs.iter().chain(ys);
        assert_eq!(iter.advance_back_by(len + 1), Err(NonZero::new(1).unwrap()));
        assert_eq!(iter.advance_back_by(0), Ok(()));
    }

    test_chain(&[], &[]);
    test_chain(&[], &[0, 1, 2, 3, 4, 5]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[30, 40, 50, 60]);
}

#[test]
fn test_iterator_chain_nth() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    for (i, x) in expected.iter().enumerate() {
        assert_eq!(Some(x), xs.iter().chain(&ys).nth(i));
    }
    assert_eq!(zs.iter().chain(&xs).nth(0), Some(&0));

    let mut it = xs.iter().chain(&zs);
    assert_eq!(it.nth(5), Some(&5));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_chain_nth_back() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    for (i, x) in expected.iter().rev().enumerate() {
        assert_eq!(Some(x), xs.iter().chain(&ys).nth_back(i));
    }
    assert_eq!(zs.iter().chain(&xs).nth_back(0), Some(&5));

    let mut it = xs.iter().chain(&zs);
    assert_eq!(it.nth_back(5), Some(&0));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_chain_last() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    assert_eq!(xs.iter().chain(&ys).last(), Some(&60));
    assert_eq!(zs.iter().chain(&ys).last(), Some(&60));
    assert_eq!(ys.iter().chain(&zs).last(), Some(&60));
    assert_eq!(zs.iter().chain(&zs).last(), None);
}

#[test]
fn test_iterator_chain_count() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    assert_eq!(xs.iter().chain(&ys).count(), 10);
    assert_eq!(zs.iter().chain(&ys).count(), 4);
}

#[test]
fn test_iterator_chain_find() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let mut iter = xs.iter().chain(&ys);
    assert_eq!(iter.find(|&&i| i == 4), Some(&4));
    assert_eq!(iter.next(), Some(&5));
    assert_eq!(iter.find(|&&i| i == 40), Some(&40));
    assert_eq!(iter.next(), Some(&50));
    assert_eq!(iter.find(|&&i| i == 100), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_iterator_chain_size_hint() {
    // this chains an iterator of length 0 with an iterator of length 1,
    // so after calling `.next()` once, the iterator is empty and the
    // state is `ChainState::Back`. `.size_hint()` should now disregard
    // the size hint of the left iterator
    let mut iter = Toggle { is_empty: true }.chain(once(()));
    assert_eq!(iter.next(), Some(()));
    assert_eq!(iter.size_hint(), (0, Some(0)));

    let mut iter = once(()).chain(Toggle { is_empty: true });
    assert_eq!(iter.next_back(), Some(()));
    assert_eq!(iter.size_hint(), (0, Some(0)));
}

#[test]
fn test_iterator_chain_unfused() {
    // Chain shouldn't be fused in its second iterator, depending on direction
    let mut iter = NonFused::new(empty()).chain(Toggle { is_empty: true });
    assert!(iter.next().is_none());
    assert!(iter.next().is_some());
    assert!(iter.next().is_none());

    let mut iter = Toggle { is_empty: true }.chain(NonFused::new(empty()));
    assert!(iter.next_back().is_none());
    assert!(iter.next_back().is_some());
    assert!(iter.next_back().is_none());
}

#[test]
fn test_chain_fold() {
    let xs = [1, 2, 3];
    let ys = [1, 2, 0];

    let mut iter = xs.iter().chain(&ys);
    iter.next();
    let mut result = Vec::new();
    iter.fold((), |(), &elt| result.push(elt));
    assert_eq!(&[2, 3, 1, 2, 0], &result[..]);
}

#[test]
fn test_chain_try_folds() {
    let c = || (0..10).chain(10..20);

    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!(c().try_fold(7, f), (0..20).try_fold(7, f));
    assert_eq!(c().try_rfold(7, f), (0..20).rev().try_fold(7, f));

    let mut iter = c();
    assert_eq!(iter.position(|x| x == 5), Some(5));
    assert_eq!(iter.next(), Some(6), "stopped in front, state Both");
    assert_eq!(iter.position(|x| x == 13), Some(6));
    assert_eq!(iter.next(), Some(14), "stopped in back, state Back");
    assert_eq!(iter.try_fold(0, |acc, x| Some(acc + x)), Some((15..20).sum()));

    let mut iter = c().rev(); // use rev to access try_rfold
    assert_eq!(iter.position(|x| x == 15), Some(4));
    assert_eq!(iter.next(), Some(14), "stopped in back, state Both");
    assert_eq!(iter.position(|x| x == 5), Some(8));
    assert_eq!(iter.next(), Some(4), "stopped in front, state Front");
    assert_eq!(iter.try_fold(0, |acc, x| Some(acc + x)), Some((0..4).sum()));

    let mut iter = c();
    iter.by_ref().rev().nth(14); // skip the last 15, ending in state Front
    assert_eq!(iter.try_fold(7, f), (0..5).try_fold(7, f));

    let mut iter = c();
    iter.nth(14); // skip the first 15, ending in state Back
    assert_eq!(iter.try_rfold(7, f), (15..20).try_rfold(7, f));
}

#[test]
fn test_double_ended_chain() {
    let xs = [1, 2, 3, 4, 5];
    let ys = [7, 9, 11];
    let mut it = xs.iter().chain(&ys).rev();
    assert_eq!(it.next().unwrap(), &11);
    assert_eq!(it.next().unwrap(), &9);
    assert_eq!(it.next_back().unwrap(), &1);
    assert_eq!(it.next_back().unwrap(), &2);
    assert_eq!(it.next_back().unwrap(), &3);
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.next_back().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);

    // test that .chain() is well behaved with an unfused iterator
    struct CrazyIterator(bool);
    impl CrazyIterator {
        fn new() -> CrazyIterator {
            CrazyIterator(false)
        }
    }
    impl Iterator for CrazyIterator {
        type Item = i32;
        fn next(&mut self) -> Option<i32> {
            if self.0 {
                Some(99)
            } else {
                self.0 = true;
                None
            }
        }
    }

    impl DoubleEndedIterator for CrazyIterator {
        fn next_back(&mut self) -> Option<i32> {
            self.next()
        }
    }

    assert_eq!(CrazyIterator::new().chain(0..10).rev().last(), Some(0));
    assert!((0..10).chain(CrazyIterator::new()).rev().any(|i| i == 0));
}

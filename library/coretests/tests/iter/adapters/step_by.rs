use core::iter::*;

#[test]
fn test_iterator_step_by() {
    // Identity
    let mut it = (0..).step_by(1).take(3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.next(), None);

    let mut it = (0..).step_by(3).take(4);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(3));
    assert_eq!(it.next(), Some(6));
    assert_eq!(it.next(), Some(9));
    assert_eq!(it.next(), None);

    let mut it = (0..3).step_by(1);
    assert_eq!(it.next_back(), Some(2));
    assert_eq!(it.next_back(), Some(1));
    assert_eq!(it.next_back(), Some(0));
    assert_eq!(it.next_back(), None);

    let mut it = (0..11).step_by(3);
    assert_eq!(it.next_back(), Some(9));
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(it.next_back(), Some(3));
    assert_eq!(it.next_back(), Some(0));
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_iterator_step_by_nth() {
    let mut it = (0..16).step_by(5);
    assert_eq!(it.nth(0), Some(0));
    assert_eq!(it.nth(0), Some(5));
    assert_eq!(it.nth(0), Some(10));
    assert_eq!(it.nth(0), Some(15));
    assert_eq!(it.nth(0), None);

    let it = (0..18).step_by(5);
    assert_eq!(it.clone().nth(0), Some(0));
    assert_eq!(it.clone().nth(1), Some(5));
    assert_eq!(it.clone().nth(2), Some(10));
    assert_eq!(it.clone().nth(3), Some(15));
    assert_eq!(it.clone().nth(4), None);
    assert_eq!(it.clone().nth(42), None);
}

#[test]
#[allow(non_local_definitions)]
fn test_iterator_step_by_nth_overflow() {
    #[cfg(target_pointer_width = "16")]
    type Bigger = u32;
    #[cfg(target_pointer_width = "32")]
    type Bigger = u64;
    #[cfg(target_pointer_width = "64")]
    type Bigger = u128;

    #[derive(Clone)]
    struct Test(Bigger);
    impl Iterator for &mut Test {
        type Item = i32;
        fn next(&mut self) -> Option<Self::Item> {
            Some(21)
        }
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.0 += n as Bigger + 1;
            Some(42)
        }
    }

    let mut it = Test(0);
    let root = usize::MAX >> (usize::BITS / 2);
    let n = root + 20;
    (&mut it).step_by(n).nth(n);
    assert_eq!(it.0, n as Bigger * n as Bigger);

    // large step
    let mut it = Test(0);
    (&mut it).step_by(usize::MAX).nth(5);
    assert_eq!(it.0, (usize::MAX as Bigger) * 5);

    // n + 1 overflows
    let mut it = Test(0);
    (&mut it).step_by(2).nth(usize::MAX);
    assert_eq!(it.0, (usize::MAX as Bigger) * 2);

    // n + 1 overflows
    let mut it = Test(0);
    (&mut it).step_by(1).nth(usize::MAX);
    assert_eq!(it.0, (usize::MAX as Bigger) * 1);
}

#[test]
fn test_iterator_step_by_nth_try_fold() {
    let mut it = (0..).step_by(10);
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it.next(), Some(60));
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it.next(), Some(90));

    let mut it = (100..).step_by(10);
    assert_eq!(it.try_fold(50, i8::checked_add), None);
    assert_eq!(it.next(), Some(110));

    let mut it = (100..=100).step_by(10);
    assert_eq!(it.next(), Some(100));
    assert_eq!(it.try_fold(0, i8::checked_add), Some(0));
}

#[test]
fn test_iterator_step_by_nth_back() {
    let mut it = (0..16).step_by(5);
    assert_eq!(it.nth_back(0), Some(15));
    assert_eq!(it.nth_back(0), Some(10));
    assert_eq!(it.nth_back(0), Some(5));
    assert_eq!(it.nth_back(0), Some(0));
    assert_eq!(it.nth_back(0), None);

    let mut it = (0..16).step_by(5);
    assert_eq!(it.next(), Some(0)); // to set `first_take` to `false`
    assert_eq!(it.nth_back(0), Some(15));
    assert_eq!(it.nth_back(0), Some(10));
    assert_eq!(it.nth_back(0), Some(5));
    assert_eq!(it.nth_back(0), None);

    let it = || (0..18).step_by(5);
    assert_eq!(it().nth_back(0), Some(15));
    assert_eq!(it().nth_back(1), Some(10));
    assert_eq!(it().nth_back(2), Some(5));
    assert_eq!(it().nth_back(3), Some(0));
    assert_eq!(it().nth_back(4), None);
    assert_eq!(it().nth_back(42), None);
}

#[test]
fn test_iterator_step_by_nth_try_rfold() {
    let mut it = (0..100).step_by(10);
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it.next_back(), Some(70));
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it.next_back(), Some(30));

    let mut it = (0..100).step_by(10);
    assert_eq!(it.try_rfold(50, i8::checked_add), None);
    assert_eq!(it.next_back(), Some(80));

    let mut it = (100..=100).step_by(10);
    assert_eq!(it.next_back(), Some(100));
    assert_eq!(it.try_fold(0, i8::checked_add), Some(0));
}

#[test]
#[should_panic]
fn test_iterator_step_by_zero() {
    let mut it = (0..).step_by(0);
    it.next();
}

#[test]
fn test_iterator_step_by_size_hint() {
    struct StubSizeHint(usize, Option<usize>);
    impl Iterator for StubSizeHint {
        type Item = ();
        fn next(&mut self) -> Option<()> {
            self.0 -= 1;
            if let Some(ref mut upper) = self.1 {
                *upper -= 1;
            }
            Some(())
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.0, self.1)
        }
    }

    // The two checks in each case are needed because the logic
    // is different before the first call to `next()`.

    let mut it = StubSizeHint(10, Some(10)).step_by(1);
    assert_eq!(it.size_hint(), (10, Some(10)));
    it.next();
    assert_eq!(it.size_hint(), (9, Some(9)));

    // exact multiple
    let mut it = StubSizeHint(10, Some(10)).step_by(3);
    assert_eq!(it.size_hint(), (4, Some(4)));
    it.next();
    assert_eq!(it.size_hint(), (3, Some(3)));

    // larger base range, but not enough to get another element
    let mut it = StubSizeHint(12, Some(12)).step_by(3);
    assert_eq!(it.size_hint(), (4, Some(4)));
    it.next();
    assert_eq!(it.size_hint(), (3, Some(3)));

    // smaller base range, so fewer resulting elements
    let mut it = StubSizeHint(9, Some(9)).step_by(3);
    assert_eq!(it.size_hint(), (3, Some(3)));
    it.next();
    assert_eq!(it.size_hint(), (2, Some(2)));

    // infinite upper bound
    let mut it = StubSizeHint(usize::MAX, None).step_by(1);
    assert_eq!(it.size_hint(), (usize::MAX, None));
    it.next();
    assert_eq!(it.size_hint(), (usize::MAX - 1, None));

    // still infinite with larger step
    let mut it = StubSizeHint(7, None).step_by(3);
    assert_eq!(it.size_hint(), (3, None));
    it.next();
    assert_eq!(it.size_hint(), (2, None));

    // propagates ExactSizeIterator
    let a = [1, 2, 3, 4, 5];
    let it = a.iter().step_by(2);
    assert_eq!(it.len(), 3);

    // Cannot be TrustedLen as a step greater than one makes an iterator
    // with (usize::MAX, None) no longer meet the safety requirements.
    // Exception: The inner iterator is known to have a len() <= usize::MAX
    trait TrustedLenCheck {
        fn test(self) -> bool;
    }
    impl<T: Iterator> TrustedLenCheck for T {
        default fn test(self) -> bool {
            false
        }
    }
    impl<T: TrustedLen> TrustedLenCheck for T {
        fn test(self) -> bool {
            true
        }
    }
    assert!(TrustedLenCheck::test(a.iter()));
    assert!(TrustedLenCheck::test(a.iter().step_by(1)));
    assert!(TrustedLenCheck::test(a.iter().chain(a.iter())));
    assert!(!TrustedLenCheck::test(a.iter().chain(a.iter()).step_by(1)));
}

#[test]
fn test_step_by_skip() {
    assert_eq!((0..640).step_by(128).skip(1).collect::<Vec<_>>(), [128, 256, 384, 512]);
    assert_eq!((0..=50).step_by(10).nth(3), Some(30));
    assert_eq!((200..=255u8).step_by(10).nth(3), Some(230));
}

struct DeOpt<I: Iterator>(I);

impl<I: Iterator> Iterator for DeOpt<I> {
    type Item = I::Item;

    fn next(&mut self) -> core::option::Option<Self::Item> {
        self.0.next()
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for DeOpt<I> {
    fn next_back(&mut self) -> core::option::Option<Self::Item> {
        self.0.next_back()
    }
}

#[test]
fn test_step_by_fold_range_specialization() {
    macro_rules! t {
        ($range:expr, $var: ident, $body:tt) => {{
            // run the same tests for the non-optimized version
            let mut $var = DeOpt($range);
            $body
        }
        {
            let mut $var = $range;
            $body
        }};
    }

    t!((1usize..5).step_by(1), r, {
        assert_eq!(r.next_back(), Some(4));
        assert_eq!(r.sum::<usize>(), 6);
    });

    t!((0usize..4).step_by(2), r, {
        assert_eq!(r.next(), Some(0));
        assert_eq!(r.sum::<usize>(), 2);
    });

    t!((0usize..5).step_by(2), r, {
        assert_eq!(r.next(), Some(0));
        assert_eq!(r.sum::<usize>(), 6);
    });

    t!((usize::MAX - 6..usize::MAX).step_by(5), r, {
        assert_eq!(r.next(), Some(usize::MAX - 6));
        assert_eq!(r.sum::<usize>(), usize::MAX - 1);
    });
}

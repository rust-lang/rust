use core::iter::*;

#[test]
fn test_cycle() {
    let cycle_len = 3;
    let it = (0..).step_by(1).take(cycle_len).cycle();
    assert_eq!(it.size_hint(), (usize::MAX, None));
    for (i, x) in it.take(100).enumerate() {
        assert_eq!(i % cycle_len, x);
    }

    let mut it = (0..).step_by(1).take(0).cycle();
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);

    assert_eq!(empty::<i32>().cycle().fold(0, |acc, x| acc + x), 0);

    assert_eq!(once(1).cycle().skip(1).take(4).fold(0, |acc, x| acc + x), 4);

    assert_eq!((0..10).cycle().take(5).sum::<i32>(), 10);
    assert_eq!((0..10).cycle().take(15).sum::<i32>(), 55);
    assert_eq!((0..10).cycle().take(25).sum::<i32>(), 100);

    let mut iter = (0..10).cycle();
    iter.nth(14);
    assert_eq!(iter.take(8).sum::<i32>(), 38);

    let mut iter = (0..10).cycle();
    iter.nth(9);
    assert_eq!(iter.take(3).sum::<i32>(), 3);
}

// Ensure that `Fuse<Cycle<I>>` behaves properly when `I` is not `FusedIterator`
// Regression test for https://github.com/rust-lang/rust/issues/166041
#[test]
fn test_cycle_fuse_1() {
    let curr = std::cell::Cell::new(0);
    let iter = std::iter::from_fn(|| {
        let n = curr.get() + 1;
        curr.set(n);
        (n >= 3).then_some("some")
    });
    let mut fused = iter.cycle().fuse();
    assert!(fused.next().is_none());
    assert!(fused.next().is_none());
    assert!(fused.next().is_none());
    assert!(fused.next().is_none());
}

// Ensure that `Fuse<Cycle<I>>` behaves properly when `I: FusedIterator`, but clones behave differently.
// Regression test for https://github.com/rust-lang/rust/pull/166122#issuecomment-6959825529
// (modified to use `Fuse<Cycle>`, since `Cycle` itself is not `FusedIterator` anymore).
#[test]
fn test_cycle_fuse_2() {
    use std::cell::Cell;
    use std::iter::FusedIterator;

    #[derive(Default)]
    pub struct ClonesEmpty(Option<i32>, Cell<usize>);
    impl Clone for ClonesEmpty {
        fn clone(&self) -> Self {
            self.1.update(|it| it + 1);
            Self((self.1.get() % 2 == 1).then_some(1), Default::default())
        }
    }
    impl Iterator for ClonesEmpty {
        type Item = i32;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.take()
        }
    }
    impl FusedIterator for ClonesEmpty {} // correctness follows from correctness for `Option::IntoIter`

    let mut idk = ClonesEmpty::default().cycle().fuse();
    assert!(idk.next().is_some());
    assert!(idk.next().is_none());
    assert!(idk.next().is_none());
    assert!(idk.next().is_none());
}

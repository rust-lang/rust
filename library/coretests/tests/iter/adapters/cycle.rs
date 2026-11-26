use core::iter::*;

#[test]
fn test_cycle() {
    let cycle_len = 3;
    let it = (0..).step_by(1).take(cycle_len).cycle();
    assert_eq!(it.size_hint(), (cycle_len, None));
    for (i, x) in it.take(100).enumerate() {
        assert_eq!(i % cycle_len, x);
    }

    let mut it = (0..).step_by(1).take(0).cycle();
    assert_eq!(it.size_hint(), (0, None));
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

// Regression test for https://github.com/rust-lang/rust/issues/166145
// `Clone` doesn't guarantee that clones yield the same items, so `Cycle::size_hint`
// must not assume the cycle is empty or infinite based on `orig`.
#[test]
fn test_cycle_size_hint_unfaithful_clone() {
    struct Custom(Option<i32>, fn() -> Self);
    impl Iterator for Custom {
        type Item = i32;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.take()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.iter().size_hint()
        }
    }
    impl Clone for Custom {
        fn clone(&self) -> Self {
            self.1()
        }
    }

    const EMPTY: Custom = Custom(None, || EMPTY);

    // The clone stored as `orig` is empty, but the live iterator isn't.
    let it = Custom(Some(1), || EMPTY).cycle();
    assert_eq!(it.size_hint(), (1, None));
    assert_eq!(it.count(), 1);

    // The clone stored as `orig` isn't empty, but the live iterator and all later clones are.
    let it = Custom(None, || Custom(Some(0), || EMPTY)).cycle();
    assert_eq!(it.size_hint(), (0, None));
    assert_eq!(it.count(), 0);
}

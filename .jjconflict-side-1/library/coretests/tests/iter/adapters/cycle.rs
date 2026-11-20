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

use core::iter::*;

#[test]
fn test_iterator_skip_while() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [15, 16, 17, 19];
    let it = xs.iter().skip_while(|&x| *x < 15);
    let mut i = 0;
    for x in it {
        assert_eq!(*x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_skip_while_fold() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [15, 16, 17, 19];
    let it = xs.iter().skip_while(|&x| *x < 15);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().skip_while(|&x| *x < 15);
    assert_eq!(it.next(), Some(&ys[0])); // process skips before folding
    let i = it.fold(1, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());
}

#[test]
fn test_skip_while_try_fold() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    fn p(&x: &i32) -> bool {
        (x % 10) <= 5
    }
    assert_eq!((1..20).skip_while(p).try_fold(7, f), (6..20).try_fold(7, f));
    let mut iter = (1..20).skip_while(p);
    assert_eq!(iter.nth(5), Some(11));
    assert_eq!(iter.try_fold(7, f), (12..20).try_fold(7, f));

    let mut iter = (0..50).skip_while(|&x| (x % 20) < 15);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(23));
}

use core::iter::*;

#[test]
fn test_iterator_take_while() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0, 1, 2, 3, 5, 13];
    let it = xs.iter().take_while(|&x| *x < 15);
    let mut i = 0;
    for x in it {
        assert_eq!(*x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_take_while_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((1..20).take_while(|&x| x != 10).try_fold(7, f), (1..10).try_fold(7, f));
    let mut iter = (1..20).take_while(|&x| x != 10);
    assert_eq!(iter.try_fold(0, |x, y| Some(x + y)), Some((1..10).sum()));
    assert_eq!(iter.next(), None, "flag should be set");
    let iter = (1..20).take_while(|&x| x != 10);
    assert_eq!(iter.fold(0, |x, y| x + y), (1..10).sum());

    let mut iter = (10..50).take_while(|&x| x != 40);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
}

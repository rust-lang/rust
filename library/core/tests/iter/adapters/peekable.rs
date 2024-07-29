use core::iter::*;

use super::*;

#[test]
fn test_iterator_peekable() {
    let xs = vec![0, 1, 2, 3, 4, 5];

    let mut it = xs.iter().cloned().peekable();
    assert_eq!(it.len(), 6);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 6);
    assert_eq!(it.next().unwrap(), 0);
    assert_eq!(it.len(), 5);
    assert_eq!(it.next().unwrap(), 1);
    assert_eq!(it.len(), 4);
    assert_eq!(it.next().unwrap(), 2);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &3);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &3);
    assert_eq!(it.len(), 3);
    assert_eq!(it.next().unwrap(), 3);
    assert_eq!(it.len(), 2);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.len(), 1);
    assert_eq!(it.peek().unwrap(), &5);
    assert_eq!(it.len(), 1);
    assert_eq!(it.next().unwrap(), 5);
    assert_eq!(it.len(), 0);
    assert!(it.peek().is_none());
    assert_eq!(it.len(), 0);
    assert!(it.next().is_none());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().cloned().peekable();
    assert_eq!(it.len(), 6);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 6);
    assert_eq!(it.next_back().unwrap(), 5);
    assert_eq!(it.len(), 5);
    assert_eq!(it.next_back().unwrap(), 4);
    assert_eq!(it.len(), 4);
    assert_eq!(it.next_back().unwrap(), 3);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 3);
    assert_eq!(it.next_back().unwrap(), 2);
    assert_eq!(it.len(), 2);
    assert_eq!(it.next_back().unwrap(), 1);
    assert_eq!(it.len(), 1);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 1);
    assert_eq!(it.next_back().unwrap(), 0);
    assert_eq!(it.len(), 0);
    assert!(it.peek().is_none());
    assert_eq!(it.len(), 0);
    assert!(it.next_back().is_none());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_peekable_count() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [10];
    let zs: [i32; 0] = [];

    assert_eq!(xs.iter().peekable().count(), 6);

    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.count(), 6);

    assert_eq!(ys.iter().peekable().count(), 1);

    let mut it = ys.iter().peekable();
    assert_eq!(it.peek(), Some(&&10));
    assert_eq!(it.count(), 1);

    assert_eq!(zs.iter().peekable().count(), 0);

    let mut it = zs.iter().peekable();
    assert_eq!(it.peek(), None);
}

#[test]
fn test_iterator_peekable_nth() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().peekable();

    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.nth(0), Some(&0));
    assert_eq!(it.peek(), Some(&&1));
    assert_eq!(it.nth(1), Some(&2));
    assert_eq!(it.peek(), Some(&&3));
    assert_eq!(it.nth(2), Some(&5));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_peekable_last() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [0];

    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.last(), Some(&5));

    let mut it = ys.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.last(), Some(&0));

    let mut it = ys.iter().peekable();
    assert_eq!(it.next(), Some(&0));
    assert_eq!(it.peek(), None);
    assert_eq!(it.last(), None);
}

#[test]
fn test_iterator_peekable_fold() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_iterator_peekable_rfold() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    let i = it.rfold(0, |i, &x| {
        assert_eq!(x, xs[xs.len() - 1 - i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_iterator_peekable_next_if_eq() {
    // first, try on references
    let xs = ["Heart", "of", "Gold"];
    let mut it = xs.into_iter().peekable();
    // try before `peek()`
    assert_eq!(it.next_if_eq(&"trillian"), None);
    assert_eq!(it.next_if_eq(&"Heart"), Some("Heart"));
    // try after peek()
    assert_eq!(it.peek(), Some(&"of"));
    assert_eq!(it.next_if_eq(&"of"), Some("of"));
    assert_eq!(it.next_if_eq(&"zaphod"), None);
    // make sure `next()` still behaves
    assert_eq!(it.next(), Some("Gold"));

    // make sure comparison works for owned values
    let xs = [String::from("Ludicrous"), "speed".into()];
    let mut it = xs.into_iter().peekable();
    // make sure basic functionality works
    assert_eq!(it.next_if_eq("Ludicrous"), Some("Ludicrous".into()));
    assert_eq!(it.next_if_eq("speed"), Some("speed".into()));
    assert_eq!(it.next_if_eq(""), None);
}

#[test]
fn test_iterator_peekable_mut() {
    let mut it = [1, 2, 3].into_iter().peekable();
    if let Some(p) = it.peek_mut() {
        if *p == 1 {
            *p = 5;
        }
    }
    assert_eq!(it.collect::<Vec<_>>(), vec![5, 2, 3]);
}

#[test]
fn test_iterator_peekable_remember_peek_none_1() {
    // Check that the loop using .peek() terminates
    let data = [1, 2, 3];
    let mut iter = CycleIter::new(&data).peekable();

    let mut n = 0;
    while let Some(_) = iter.next() {
        let is_the_last = iter.peek().is_none();
        assert_eq!(is_the_last, n == data.len() - 1);
        n += 1;
        if n > data.len() {
            break;
        }
    }
    assert_eq!(n, data.len());
}

#[test]
fn test_iterator_peekable_remember_peek_none_2() {
    let data = [0];
    let mut iter = CycleIter::new(&data).peekable();
    iter.next();
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.last(), None);
}

#[test]
fn test_iterator_peekable_remember_peek_none_3() {
    let data = [0];
    let mut iter = CycleIter::new(&data).peekable();
    iter.peek();
    assert_eq!(iter.nth(0), Some(&0));

    let mut iter = CycleIter::new(&data).peekable();
    iter.next();
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.nth(0), None);
}

#[test]
fn test_peek_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);

    assert_eq!((1..20).peekable().try_fold(7, f), (1..20).try_fold(7, f));
    assert_eq!((1..20).peekable().try_rfold(7, f), (1..20).try_rfold(7, f));

    let mut iter = (1..20).peekable();
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.try_fold(7, f), (1..20).try_fold(7, f));

    let mut iter = (1..20).peekable();
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.try_rfold(7, f), (1..20).try_rfold(7, f));

    let mut iter = [100, 20, 30, 40, 50, 60, 70].iter().cloned().peekable();
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.peek(), Some(&40));

    let mut iter = [100, 20, 30, 40, 50, 60, 70].iter().cloned().peekable();
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.next_back(), Some(50));

    let mut iter = (2..5).peekable();
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.peek(), Some(&3));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.peek(), Some(&4));
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.try_for_each(Err), Ok(()));

    let mut iter = (2..5).peekable();
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Err(4));
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Err(3));
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Err(2));
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Ok(()));
}

#[test]
fn test_peekable_non_fused() {
    let mut iter = NonFused::new(empty::<i32>()).peekable();

    assert_eq!(iter.peek(), None);
    assert_eq!(iter.next_back(), None);
}

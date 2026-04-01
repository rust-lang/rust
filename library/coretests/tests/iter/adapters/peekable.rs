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

#[test]
fn test_peekable_next_if_map_mutation() {
    fn collatz((mut num, mut len): (u64, u32)) -> Result<u32, (u64, u32)> {
        let jump = num.trailing_zeros();
        num >>= jump;
        len += jump;
        if num == 1 { Ok(len) } else { Err((3 * num + 1, len + 1)) }
    }

    let mut iter = once((3, 0)).peekable();
    assert_eq!(iter.peek(), Some(&(3, 0)));
    assert_eq!(iter.next_if_map(collatz), None);
    assert_eq!(iter.peek(), Some(&(10, 1)));
    assert_eq!(iter.next_if_map(collatz), None);
    assert_eq!(iter.peek(), Some(&(16, 3)));
    assert_eq!(iter.next_if_map(collatz), Some(7));
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.next_if_map(collatz), None);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_peekable_next_if_map_panic() {
    use core::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct BitsetOnDrop<'a> {
        value: u32,
        cell: &'a Cell<u32>,
    }
    impl<'a> Drop for BitsetOnDrop<'a> {
        fn drop(&mut self) {
            self.cell.update(|v| v | self.value);
        }
    }

    let cell = &Cell::new(0);
    let mut it = [
        BitsetOnDrop { value: 1, cell },
        BitsetOnDrop { value: 2, cell },
        BitsetOnDrop { value: 4, cell },
        BitsetOnDrop { value: 8, cell },
    ]
    .into_iter()
    .peekable();

    // sanity check, .peek() won't consume the value, .next() will transfer ownership.
    let item = it.peek().unwrap();
    assert_eq!(item.value, 1);
    assert_eq!(cell.get(), 0);
    let item = it.next().unwrap();
    assert_eq!(item.value, 1);
    assert_eq!(cell.get(), 0);
    drop(item);
    assert_eq!(cell.get(), 1);

    // next_if_map returning Ok should transfer the value out.
    let item = it.next_if_map(Ok).unwrap();
    assert_eq!(item.value, 2);
    assert_eq!(cell.get(), 1);
    drop(item);
    assert_eq!(cell.get(), 3);

    // next_if_map returning Err should not drop anything.
    assert_eq!(it.next_if_map::<()>(Err), None);
    assert_eq!(cell.get(), 3);
    assert_eq!(it.peek().unwrap().value, 4);
    assert_eq!(cell.get(), 3);

    // next_if_map panicking should consume and drop the item.
    let result = catch_unwind({
        let mut it = AssertUnwindSafe(&mut it);
        move || it.next_if_map::<()>(|_| panic!())
    });
    assert!(result.is_err());
    assert_eq!(cell.get(), 7);
    assert_eq!(it.next().unwrap().value, 8);
    assert_eq!(cell.get(), 15);
    assert!(it.peek().is_none());

    // next_if_map should *not* execute the closure if the iterator is exhausted.
    assert!(it.next_if_map::<()>(|_| panic!()).is_none());
    assert!(it.peek().is_none());
    assert_eq!(cell.get(), 15);
}

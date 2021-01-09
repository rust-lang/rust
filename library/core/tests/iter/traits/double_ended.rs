#[test]
fn test_iterator_rev_nth_back() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().rev().nth_back(i).unwrap(), &v[i]);
    }
    assert_eq!(v.iter().rev().nth_back(v.len()), None);
}
#[test]
fn test_iterator_rev_nth() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().rev().nth(i).unwrap(), &v[v.len() - 1 - i]);
    }
    assert_eq!(v.iter().rev().nth(v.len()), None);
}
#[test]
fn test_iterator_len() {
    let v: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().count(), 4);
    assert_eq!(v[..10].iter().count(), 10);
    assert_eq!(v[..0].iter().count(), 0);
}
#[test]
fn test_rev() {
    let xs = [2, 4, 6, 8, 10, 12, 14, 16];
    let mut it = xs.iter();
    it.next();
    it.next();
    assert!(it.rev().cloned().collect::<Vec<isize>>() == vec![16, 14, 12, 10, 8, 6]);
}
#[test]
fn test_double_ended_map() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().map(|&x| x * -1);
    assert_eq!(it.next(), Some(-1));
    assert_eq!(it.next(), Some(-2));
    assert_eq!(it.next_back(), Some(-6));
    assert_eq!(it.next_back(), Some(-5));
    assert_eq!(it.next(), Some(-3));
    assert_eq!(it.next_back(), Some(-4));
    assert_eq!(it.next(), None);
}
#[test]
fn test_double_ended_enumerate() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().cloned().enumerate();
    assert_eq!(it.next(), Some((0, 1)));
    assert_eq!(it.next(), Some((1, 2)));
    assert_eq!(it.next_back(), Some((5, 6)));
    assert_eq!(it.next_back(), Some((4, 5)));
    assert_eq!(it.next_back(), Some((3, 4)));
    assert_eq!(it.next_back(), Some((2, 3)));
    assert_eq!(it.next(), None);
}
#[test]
fn test_double_ended_zip() {
    let xs = [1, 2, 3, 4, 5, 6];
    let ys = [1, 2, 3, 7];
    let a = xs.iter().cloned();
    let b = ys.iter().cloned();
    let mut it = a.zip(b);
    assert_eq!(it.next(), Some((1, 1)));
    assert_eq!(it.next(), Some((2, 2)));
    assert_eq!(it.next_back(), Some((4, 7)));
    assert_eq!(it.next_back(), Some((3, 3)));
    assert_eq!(it.next(), None);
}
#[test]
fn test_double_ended_filter() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter(|&x| *x & 1 == 0);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.next().unwrap(), &2);
    assert_eq!(it.next_back(), None);
}
#[test]
fn test_double_ended_filter_map() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter_map(|&x| if x & 1 == 0 { Some(x * 2) } else { None });
    assert_eq!(it.next_back().unwrap(), 12);
    assert_eq!(it.next_back().unwrap(), 8);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.next_back(), None);
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
#[test]
fn test_double_ended_flat_map() {
    let u = [0, 1];
    let v = [5, 6, 7, 8];
    let mut it = u.iter().flat_map(|x| &v[*x..v.len()]);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}
#[test]
fn test_double_ended_flatten() {
    let u = [0, 1];
    let v = [5, 6, 7, 8];
    let mut it = u.iter().map(|x| &v[*x..v.len()]).flatten();
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}
#[test]
fn test_double_ended_range() {
    assert_eq!((11..14).rev().collect::<Vec<_>>(), [13, 12, 11]);
    for _ in (10..0).rev() {
        panic!("unreachable");
    }

    assert_eq!((11..14).rev().collect::<Vec<_>>(), [13, 12, 11]);
    for _ in (10..0).rev() {
        panic!("unreachable");
    }
}
#[test]
fn test_rev_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((1..10).rev().try_fold(7, f), (1..10).try_rfold(7, f));
    assert_eq!((1..10).rev().try_rfold(7, f), (1..10).try_fold(7, f));

    let a = [10, 20, 30, 40, 100, 60, 70, 80, 90];
    let mut iter = a.iter().rev();
    assert_eq!(iter.try_fold(0_i8, |acc, &x| acc.checked_add(x)), None);
    assert_eq!(iter.next(), Some(&70));
    let mut iter = a.iter().rev();
    assert_eq!(iter.try_rfold(0_i8, |acc, &x| acc.checked_add(x)), None);
    assert_eq!(iter.next_back(), Some(&60));
}

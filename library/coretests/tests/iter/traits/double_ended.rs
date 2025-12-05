//! Note
//! ----
//! You're probably viewing this file because you're adding a test (or you might
//! just be browsing, in that case, hey there!).
//!
//! If you've made a test that happens to use one of DoubleEnded's methods, but
//! it tests another adapter or trait, you should *add it to the adapter or
//! trait's test file*.
//!
//! Some examples would be `adapters::cloned::test_cloned_try_folds` or
//! `adapters::flat_map::test_double_ended_flat_map`, which use `try_fold` and
//! `next_back`, but test their own adapter.

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
fn test_rev() {
    let xs = [2, 4, 6, 8, 10, 12, 14, 16];
    let mut it = xs.iter();
    it.next();
    it.next();
    assert!(it.rev().cloned().collect::<Vec<isize>>() == vec![16, 14, 12, 10, 8, 6]);
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

#[test]
fn test_rposition() {
    fn f(xy: &(isize, char)) -> bool {
        let (_x, y) = *xy;
        y == 'b'
    }
    fn g(xy: &(isize, char)) -> bool {
        let (_x, y) = *xy;
        y == 'd'
    }
    let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

    assert_eq!(v.iter().rposition(f), Some(3));
    assert!(v.iter().rposition(g).is_none());
}

#[test]
fn test_rev_rposition() {
    let v = [0, 0, 1, 1];
    assert_eq!(v.iter().rev().rposition(|&x| x == 1), Some(1));
}

#[test]
#[should_panic]
fn test_rposition_panic() {
    let u = (Box::new(0), Box::new(0));
    let v: [(Box<_>, Box<_>); 4] = [u.clone(), u.clone(), u.clone(), u];
    let mut i = 0;
    v.iter().rposition(|_elt| {
        if i == 2 {
            panic!()
        }
        i += 1;
        false
    });
}

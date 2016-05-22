// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::LinkedList;

use test;

#[test]
fn test_basic() {
    let mut m = LinkedList::<Box<_>>::new();
    assert_eq!(m.pop_front(), None);
    assert_eq!(m.pop_back(), None);
    assert_eq!(m.pop_front(), None);
    m.push_front(box 1);
    assert_eq!(m.pop_front(), Some(box 1));
    m.push_back(box 2);
    m.push_back(box 3);
    assert_eq!(m.len(), 2);
    assert_eq!(m.pop_front(), Some(box 2));
    assert_eq!(m.pop_front(), Some(box 3));
    assert_eq!(m.len(), 0);
    assert_eq!(m.pop_front(), None);
    m.push_back(box 1);
    m.push_back(box 3);
    m.push_back(box 5);
    m.push_back(box 7);
    assert_eq!(m.pop_front(), Some(box 1));

    let mut n = LinkedList::new();
    n.push_front(2);
    n.push_front(3);
    {
        assert_eq!(n.front().unwrap(), &3);
        let x = n.front_mut().unwrap();
        assert_eq!(*x, 3);
        *x = 0;
    }
    {
        assert_eq!(n.back().unwrap(), &2);
        let y = n.back_mut().unwrap();
        assert_eq!(*y, 2);
        *y = 1;
    }
    assert_eq!(n.pop_front(), Some(0));
    assert_eq!(n.pop_front(), Some(1));
}

#[cfg(test)]
fn generate_test() -> LinkedList<i32> {
    list_from(&[0, 1, 2, 3, 4, 5, 6])
}

#[cfg(test)]
fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
    v.iter().cloned().collect()
}

#[test]
fn test_split_off() {
    // singleton
    {
        let mut m = LinkedList::new();
        m.push_back(1);

        let p = m.split_off(0);
        assert_eq!(m.len(), 0);
        assert_eq!(p.len(), 1);
        assert_eq!(p.back(), Some(&1));
        assert_eq!(p.front(), Some(&1));
    }

    // not singleton, forwards
    {
        let u = vec![1, 2, 3, 4, 5];
        let mut m = list_from(&u);
        let mut n = m.split_off(2);
        assert_eq!(m.len(), 2);
        assert_eq!(n.len(), 3);
        for elt in 1..3 {
            assert_eq!(m.pop_front(), Some(elt));
        }
        for elt in 3..6 {
            assert_eq!(n.pop_front(), Some(elt));
        }
    }
    // not singleton, backwards
    {
        let u = vec![1, 2, 3, 4, 5];
        let mut m = list_from(&u);
        let mut n = m.split_off(4);
        assert_eq!(m.len(), 4);
        assert_eq!(n.len(), 1);
        for elt in 1..5 {
            assert_eq!(m.pop_front(), Some(elt));
        }
        for elt in 5..6 {
            assert_eq!(n.pop_front(), Some(elt));
        }
    }

    // no-op on the last index
    {
        let mut m = LinkedList::new();
        m.push_back(1);

        let p = m.split_off(1);
        assert_eq!(m.len(), 1);
        assert_eq!(p.len(), 0);
        assert_eq!(m.back(), Some(&1));
        assert_eq!(m.front(), Some(&1));
    }

}

#[test]
fn test_iterator() {
    let m = generate_test();
    for (i, elt) in m.iter().enumerate() {
        assert_eq!(i as i32, *elt);
    }
    let mut n = LinkedList::new();
    assert_eq!(n.iter().next(), None);
    n.push_front(4);
    let mut it = n.iter();
    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next().unwrap(), &4);
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_clone() {
    let mut n = LinkedList::new();
    n.push_back(2);
    n.push_back(3);
    n.push_back(4);
    let mut it = n.iter();
    it.next();
    let mut jt = it.clone();
    assert_eq!(it.next(), jt.next());
    assert_eq!(it.next_back(), jt.next_back());
    assert_eq!(it.next(), jt.next());
}

#[test]
fn test_iterator_double_end() {
    let mut n = LinkedList::new();
    assert_eq!(n.iter().next(), None);
    n.push_front(4);
    n.push_front(5);
    n.push_front(6);
    let mut it = n.iter();
    assert_eq!(it.size_hint(), (3, Some(3)));
    assert_eq!(it.next().unwrap(), &6);
    assert_eq!(it.size_hint(), (2, Some(2)));
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next_back().unwrap(), &5);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
}

#[test]
fn test_rev_iter() {
    let m = generate_test();
    for (i, elt) in m.iter().rev().enumerate() {
        assert_eq!((6 - i) as i32, *elt);
    }
    let mut n = LinkedList::new();
    assert_eq!(n.iter().rev().next(), None);
    n.push_front(4);
    let mut it = n.iter().rev();
    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next().unwrap(), &4);
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_mut_iter() {
    let mut m = generate_test();
    let mut len = m.len();
    for (i, elt) in m.iter_mut().enumerate() {
        assert_eq!(i as i32, *elt);
        len -= 1;
    }
    assert_eq!(len, 0);
    let mut n = LinkedList::new();
    assert!(n.iter_mut().next().is_none());
    n.push_front(4);
    n.push_back(5);
    let mut it = n.iter_mut();
    assert_eq!(it.size_hint(), (2, Some(2)));
    assert!(it.next().is_some());
    assert!(it.next().is_some());
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
}

#[test]
fn test_iterator_mut_double_end() {
    let mut n = LinkedList::new();
    assert!(n.iter_mut().next_back().is_none());
    n.push_front(4);
    n.push_front(5);
    n.push_front(6);
    let mut it = n.iter_mut();
    assert_eq!(it.size_hint(), (3, Some(3)));
    assert_eq!(*it.next().unwrap(), 6);
    assert_eq!(it.size_hint(), (2, Some(2)));
    assert_eq!(*it.next_back().unwrap(), 4);
    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(*it.next_back().unwrap(), 5);
    assert!(it.next_back().is_none());
    assert!(it.next().is_none());
}

#[test]
fn test_mut_rev_iter() {
    let mut m = generate_test();
    for (i, elt) in m.iter_mut().rev().enumerate() {
        assert_eq!((6 - i) as i32, *elt);
    }
    let mut n = LinkedList::new();
    assert!(n.iter_mut().rev().next().is_none());
    n.push_front(4);
    let mut it = n.iter_mut().rev();
    assert!(it.next().is_some());
    assert!(it.next().is_none());
}

#[test]
fn test_eq() {
    let mut n = list_from(&[]);
    let mut m = list_from(&[]);
    assert!(n == m);
    n.push_front(1);
    assert!(n != m);
    m.push_back(1);
    assert!(n == m);

    let n = list_from(&[2, 3, 4]);
    let m = list_from(&[1, 2, 3]);
    assert!(n != m);
}

#[test]
fn test_hash() {
    let mut x = LinkedList::new();
    let mut y = LinkedList::new();

    assert!(::hash(&x) == ::hash(&y));

    x.push_back(1);
    x.push_back(2);
    x.push_back(3);

    y.push_front(3);
    y.push_front(2);
    y.push_front(1);

    assert!(::hash(&x) == ::hash(&y));
}

#[test]
fn test_ord() {
    let n = list_from(&[]);
    let m = list_from(&[1, 2, 3]);
    assert!(n < m);
    assert!(m > n);
    assert!(n <= n);
    assert!(n >= n);
}

#[test]
fn test_ord_nan() {
    let nan = 0.0f64 / 0.0;
    let n = list_from(&[nan]);
    let m = list_from(&[nan]);
    assert!(!(n < m));
    assert!(!(n > m));
    assert!(!(n <= m));
    assert!(!(n >= m));

    let n = list_from(&[nan]);
    let one = list_from(&[1.0f64]);
    assert!(!(n < one));
    assert!(!(n > one));
    assert!(!(n <= one));
    assert!(!(n >= one));

    let u = list_from(&[1.0f64, 2.0, nan]);
    let v = list_from(&[1.0f64, 2.0, 3.0]);
    assert!(!(u < v));
    assert!(!(u > v));
    assert!(!(u <= v));
    assert!(!(u >= v));

    let s = list_from(&[1.0f64, 2.0, 4.0, 2.0]);
    let t = list_from(&[1.0f64, 2.0, 3.0, 2.0]);
    assert!(!(s < t));
    assert!(s > one);
    assert!(!(s <= one));
    assert!(s >= one);
}

#[test]
fn test_show() {
    let list: LinkedList<_> = (0..10).collect();
    assert_eq!(format!("{:?}", list), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

    let list: LinkedList<_> = vec!["just", "one", "test", "more"].iter().cloned().collect();
    assert_eq!(format!("{:?}", list),
               "[\"just\", \"one\", \"test\", \"more\"]");
}

#[test]
fn test_extend_ref() {
    let mut a = LinkedList::new();
    a.push_back(1);

    a.extend(&[2, 3, 4]);

    assert_eq!(a.len(), 4);
    assert_eq!(a, list_from(&[1, 2, 3, 4]));

    let mut b = LinkedList::new();
    b.push_back(5);
    b.push_back(6);
    a.extend(&b);

    assert_eq!(a.len(), 6);
    assert_eq!(a, list_from(&[1, 2, 3, 4, 5, 6]));
}

#[test]
fn test_extend() {
    let mut a = LinkedList::new();
    a.push_back(1);
    a.extend(vec![2, 3, 4]); // uses iterator

    assert_eq!(a.len(), 4);
    assert!(a.iter().eq(&[1, 2, 3, 4]));

    let b: LinkedList<_> = vec![5, 6, 7].into_iter().collect();
    a.extend(b); // specializes to `append`

    assert_eq!(a.len(), 7);
    assert!(a.iter().eq(&[1, 2, 3, 4, 5, 6, 7]));
}

#[bench]
fn bench_collect_into(b: &mut test::Bencher) {
    let v = &[0; 64];
    b.iter(|| {
        let _: LinkedList<_> = v.iter().cloned().collect();
    })
}

#[bench]
fn bench_push_front(b: &mut test::Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_front(0);
    })
}

#[bench]
fn bench_push_back(b: &mut test::Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_back(0);
    })
}

#[bench]
fn bench_push_back_pop_back(b: &mut test::Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_back(0);
        m.pop_back();
    })
}

#[bench]
fn bench_push_front_pop_front(b: &mut test::Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_front(0);
        m.pop_front();
    })
}

#[bench]
fn bench_iter(b: &mut test::Bencher) {
    let v = &[0; 128];
    let m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter().count() == 128);
    })
}
#[bench]
fn bench_iter_mut(b: &mut test::Bencher) {
    let v = &[0; 128];
    let mut m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter_mut().count() == 128);
    })
}
#[bench]
fn bench_iter_rev(b: &mut test::Bencher) {
    let v = &[0; 128];
    let m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter().rev().count() == 128);
    })
}
#[bench]
fn bench_iter_mut_rev(b: &mut test::Bencher) {
    let v = &[0; 128];
    let mut m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter_mut().rev().count() == 128);
    })
}

#[test]
fn test_contains() {
    let mut l = LinkedList::new();
    l.extend(&[2, 3, 4]);

    assert!(l.contains(&3));
    assert!(!l.contains(&1));

    l.clear();

    assert!(!l.contains(&3));
}

use std::collections::LinkedList;
use std::panic::{catch_unwind, AssertUnwindSafe};

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

fn generate_test() -> LinkedList<i32> {
    list_from(&[0, 1, 2, 3, 4, 5, 6])
}

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
    use crate::hash;

    let mut x = LinkedList::new();
    let mut y = LinkedList::new();

    assert!(hash(&x) == hash(&y));

    x.push_back(1);
    x.push_back(2);
    x.push_back(3);

    y.push_front(3);
    y.push_front(2);
    y.push_front(1);

    assert!(hash(&x) == hash(&y));
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

    let list: LinkedList<_> = ["just", "one", "test", "more"].into_iter().collect();
    assert_eq!(format!("{:?}", list), "[\"just\", \"one\", \"test\", \"more\"]");
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

    let b: LinkedList<_> = [5, 6, 7].into_iter().collect();
    a.extend(b); // specializes to `append`

    assert_eq!(a.len(), 7);
    assert!(a.iter().eq(&[1, 2, 3, 4, 5, 6, 7]));
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

#[test]
fn drain_filter_empty() {
    let mut list: LinkedList<i32> = LinkedList::new();

    {
        let mut iter = list.drain_filter(|_| true);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(list.len(), 0);
    assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn drain_filter_zst() {
    let mut list: LinkedList<_> = [(), (), (), (), ()].into_iter().collect();
    let initial_len = list.len();
    let mut count = 0;

    {
        let mut iter = list.drain_filter(|_| true);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        while let Some(_) = iter.next() {
            count += 1;
            assert_eq!(iter.size_hint(), (0, Some(initial_len - count)));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, initial_len);
    assert_eq!(list.len(), 0);
    assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn drain_filter_false() {
    let mut list: LinkedList<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

    let initial_len = list.len();
    let mut count = 0;

    {
        let mut iter = list.drain_filter(|_| false);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        for _ in iter.by_ref() {
            count += 1;
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, 0);
    assert_eq!(list.len(), initial_len);
    assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn drain_filter_true() {
    let mut list: LinkedList<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

    let initial_len = list.len();
    let mut count = 0;

    {
        let mut iter = list.drain_filter(|_| true);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        while let Some(_) = iter.next() {
            count += 1;
            assert_eq!(iter.size_hint(), (0, Some(initial_len - count)));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, initial_len);
    assert_eq!(list.len(), 0);
    assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn drain_filter_complex() {
    {
        //                [+xxx++++++xxxxx++++x+x++]
        let mut list = [
            1, 2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36, 37,
            39,
        ]
        .into_iter()
        .collect::<LinkedList<_>>();

        let removed = list.drain_filter(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(list.len(), 14);
        assert_eq!(
            list.into_iter().collect::<Vec<_>>(),
            vec![1, 7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39]
        );
    }

    {
        // [xxx++++++xxxxx++++x+x++]
        let mut list =
            [2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36, 37, 39]
                .into_iter()
                .collect::<LinkedList<_>>();

        let removed = list.drain_filter(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(list.len(), 13);
        assert_eq!(
            list.into_iter().collect::<Vec<_>>(),
            vec![7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39]
        );
    }

    {
        // [xxx++++++xxxxx++++x+x]
        let mut list =
            [2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36]
                .into_iter()
                .collect::<LinkedList<_>>();

        let removed = list.drain_filter(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(list.len(), 11);
        assert_eq!(
            list.into_iter().collect::<Vec<_>>(),
            vec![7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35]
        );
    }

    {
        // [xxxxxxxxxx+++++++++++]
        let mut list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            .into_iter()
            .collect::<LinkedList<_>>();

        let removed = list.drain_filter(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(list.len(), 10);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }

    {
        // [+++++++++++xxxxxxxxxx]
        let mut list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            .into_iter()
            .collect::<LinkedList<_>>();

        let removed = list.drain_filter(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(list.len(), 10);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }
}

#[test]
fn drain_filter_drop_panic_leak() {
    static mut DROPS: i32 = 0;

    struct D(bool);

    impl Drop for D {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }

            if self.0 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut q = LinkedList::new();
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_front(D(false));
    q.push_front(D(true));
    q.push_front(D(false));

    catch_unwind(AssertUnwindSafe(|| drop(q.drain_filter(|_| true)))).ok();

    assert_eq!(unsafe { DROPS }, 8);
    assert!(q.is_empty());
}

#[test]
fn drain_filter_pred_panic_leak() {
    static mut DROPS: i32 = 0;

    #[derive(Debug)]
    struct D(u32);

    impl Drop for D {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let mut q = LinkedList::new();
    q.push_back(D(3));
    q.push_back(D(4));
    q.push_back(D(5));
    q.push_back(D(6));
    q.push_back(D(7));
    q.push_front(D(2));
    q.push_front(D(1));
    q.push_front(D(0));

    catch_unwind(AssertUnwindSafe(|| {
        drop(q.drain_filter(|item| if item.0 >= 2 { panic!() } else { true }))
    }))
    .ok();

    assert_eq!(unsafe { DROPS }, 2); // 0 and 1
    assert_eq!(q.len(), 6);
}

#[test]
fn test_drop() {
    static mut DROPS: i32 = 0;
    struct Elem;
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let mut ring = LinkedList::new();
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.push_back(Elem);
    ring.push_front(Elem);
    drop(ring);

    assert_eq!(unsafe { DROPS }, 4);
}

#[test]
fn test_drop_with_pop() {
    static mut DROPS: i32 = 0;
    struct Elem;
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let mut ring = LinkedList::new();
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.push_back(Elem);
    ring.push_front(Elem);

    drop(ring.pop_back());
    drop(ring.pop_front());
    assert_eq!(unsafe { DROPS }, 2);

    drop(ring);
    assert_eq!(unsafe { DROPS }, 4);
}

#[test]
fn test_drop_clear() {
    static mut DROPS: i32 = 0;
    struct Elem;
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let mut ring = LinkedList::new();
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.clear();
    assert_eq!(unsafe { DROPS }, 4);

    drop(ring);
    assert_eq!(unsafe { DROPS }, 4);
}

#[test]
fn test_drop_panic() {
    static mut DROPS: i32 = 0;

    struct D(bool);

    impl Drop for D {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }

            if self.0 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut q = LinkedList::new();
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_front(D(false));
    q.push_front(D(false));
    q.push_front(D(true));

    catch_unwind(move || drop(q)).ok();

    assert_eq!(unsafe { DROPS }, 8);
}

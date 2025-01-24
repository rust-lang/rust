// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;

use rand::RngCore;

use super::*;
use crate::testing::crash_test::{CrashTestDummy, Panic};
use crate::vec::Vec;

#[test]
fn test_basic() {
    let mut m = LinkedList::<Box<_>>::new();
    assert_eq!(m.pop_front(), None);
    assert_eq!(m.pop_back(), None);
    assert_eq!(m.pop_front(), None);
    m.push_front(Box::new(1));
    assert_eq!(m.pop_front(), Some(Box::new(1)));
    m.push_back(Box::new(2));
    m.push_back(Box::new(3));
    assert_eq!(m.len(), 2);
    assert_eq!(m.pop_front(), Some(Box::new(2)));
    assert_eq!(m.pop_front(), Some(Box::new(3)));
    assert_eq!(m.len(), 0);
    assert_eq!(m.pop_front(), None);
    m.push_back(Box::new(1));
    m.push_back(Box::new(3));
    m.push_back(Box::new(5));
    m.push_back(Box::new(7));
    assert_eq!(m.pop_front(), Some(Box::new(1)));

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

pub fn check_links<T>(list: &LinkedList<T>) {
    unsafe {
        let mut len = 0;
        let mut last_ptr: Option<&Node<T>> = None;
        let mut node_ptr: &Node<T>;
        match list.head {
            None => {
                // tail node should also be None.
                assert!(list.tail.is_none());
                assert_eq!(0, list.len);
                return;
            }
            Some(node) => node_ptr = &*node.as_ptr(),
        }
        loop {
            match (last_ptr, node_ptr.prev) {
                (None, None) => {}
                (None, _) => panic!("prev link for head"),
                (Some(p), Some(pptr)) => {
                    assert_eq!(p as *const Node<T>, pptr.as_ptr() as *const Node<T>);
                }
                _ => panic!("prev link is none, not good"),
            }
            match node_ptr.next {
                Some(next) => {
                    last_ptr = Some(node_ptr);
                    node_ptr = &*next.as_ptr();
                    len += 1;
                }
                None => {
                    len += 1;
                    break;
                }
            }
        }

        // verify that the tail node points to the last node.
        let tail = list.tail.as_ref().expect("some tail node").as_ref();
        assert_eq!(tail as *const Node<T>, node_ptr as *const Node<T>);
        // check that len matches interior links.
        assert_eq!(len, list.len);
    }
}

#[test]
fn test_append() {
    // Empty to empty
    {
        let mut m = LinkedList::<i32>::new();
        let mut n = LinkedList::new();
        m.append(&mut n);
        check_links(&m);
        assert_eq!(m.len(), 0);
        assert_eq!(n.len(), 0);
    }
    // Non-empty to empty
    {
        let mut m = LinkedList::new();
        let mut n = LinkedList::new();
        n.push_back(2);
        m.append(&mut n);
        check_links(&m);
        assert_eq!(m.len(), 1);
        assert_eq!(m.pop_back(), Some(2));
        assert_eq!(n.len(), 0);
        check_links(&m);
    }
    // Empty to non-empty
    {
        let mut m = LinkedList::new();
        let mut n = LinkedList::new();
        m.push_back(2);
        m.append(&mut n);
        check_links(&m);
        assert_eq!(m.len(), 1);
        assert_eq!(m.pop_back(), Some(2));
        check_links(&m);
    }

    // Non-empty to non-empty
    let v = vec![1, 2, 3, 4, 5];
    let u = vec![9, 8, 1, 2, 3, 4, 5];
    let mut m = list_from(&v);
    let mut n = list_from(&u);
    m.append(&mut n);
    check_links(&m);
    let mut sum = v;
    sum.extend_from_slice(&u);
    assert_eq!(sum.len(), m.len());
    for elt in sum {
        assert_eq!(m.pop_front(), Some(elt))
    }
    assert_eq!(n.len(), 0);
    // Let's make sure it's working properly, since we
    // did some direct changes to private members.
    n.push_back(3);
    assert_eq!(n.len(), 1);
    assert_eq!(n.pop_front(), Some(3));
    check_links(&n);
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
fn test_clone_from() {
    // Short cloned from long
    {
        let v = vec![1, 2, 3, 4, 5];
        let u = vec![8, 7, 6, 2, 3, 4, 5];
        let mut m = list_from(&v);
        let n = list_from(&u);
        m.clone_from(&n);
        check_links(&m);
        assert_eq!(m, n);
        for elt in u {
            assert_eq!(m.pop_front(), Some(elt))
        }
    }
    // Long cloned from short
    {
        let v = vec![1, 2, 3, 4, 5];
        let u = vec![6, 7, 8];
        let mut m = list_from(&v);
        let n = list_from(&u);
        m.clone_from(&n);
        check_links(&m);
        assert_eq!(m, n);
        for elt in u {
            assert_eq!(m.pop_front(), Some(elt))
        }
    }
    // Two equal length lists
    {
        let v = vec![1, 2, 3, 4, 5];
        let u = vec![9, 8, 1, 2, 3];
        let mut m = list_from(&v);
        let n = list_from(&u);
        m.clone_from(&n);
        check_links(&m);
        assert_eq!(m, n);
        for elt in u {
            assert_eq!(m.pop_front(), Some(elt))
        }
    }
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_send() {
    let n = list_from(&[1, 2, 3]);
    thread::spawn(move || {
        check_links(&n);
        let a: &[_] = &[&1, &2, &3];
        assert_eq!(a, &*n.iter().collect::<Vec<_>>());
    })
    .join()
    .ok()
    .unwrap();
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
fn test_26021() {
    // There was a bug in split_off that failed to null out the RHS's head's prev ptr.
    // This caused the RHS's dtor to walk up into the LHS at drop and delete all of
    // its nodes.
    //
    // https://github.com/rust-lang/rust/issues/26021
    let mut v1 = LinkedList::new();
    v1.push_front(1);
    v1.push_front(1);
    v1.push_front(1);
    v1.push_front(1);
    let _ = v1.split_off(3); // Dropping this now should not cause laundry consumption
    assert_eq!(v1.len(), 3);

    assert_eq!(v1.iter().len(), 3);
    assert_eq!(v1.iter().collect::<Vec<_>>().len(), 3);
}

#[test]
fn test_split_off() {
    let mut v1 = LinkedList::new();
    v1.push_front(1);
    v1.push_front(1);
    v1.push_front(1);
    v1.push_front(1);

    // test all splits
    for ix in 0..1 + v1.len() {
        let mut a = v1.clone();
        let b = a.split_off(ix);
        check_links(&a);
        check_links(&b);
        a.extend(b);
        assert_eq!(v1, a);
    }
}

#[test]
fn test_split_off_2() {
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

fn fuzz_test(sz: i32, rng: &mut impl RngCore) {
    let mut m: LinkedList<_> = LinkedList::new();
    let mut v = vec![];
    for i in 0..sz {
        check_links(&m);
        let r: u8 = rng.next_u32() as u8;
        match r % 6 {
            0 => {
                m.pop_back();
                v.pop();
            }
            1 => {
                if !v.is_empty() {
                    m.pop_front();
                    v.remove(0);
                }
            }
            2 | 4 => {
                m.push_front(-i);
                v.insert(0, -i);
            }
            3 | 5 | _ => {
                m.push_back(i);
                v.push(i);
            }
        }
    }

    check_links(&m);

    let mut i = 0;
    for (a, &b) in m.into_iter().zip(&v) {
        i += 1;
        assert_eq!(a, b);
    }
    assert_eq!(i, v.len());
}

#[test]
fn test_fuzz() {
    let mut rng = crate::test_helpers::test_rng();
    for _ in 0..25 {
        fuzz_test(3, &mut rng);
        fuzz_test(16, &mut rng);
        #[cfg(not(miri))] // Miri is too slow
        fuzz_test(189, &mut rng);
    }
}

#[test]
fn test_show() {
    let list: LinkedList<_> = (0..10).collect();
    assert_eq!(format!("{list:?}"), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

    let list: LinkedList<_> = ["just", "one", "test", "more"].into_iter().collect();
    assert_eq!(format!("{list:?}"), "[\"just\", \"one\", \"test\", \"more\"]");
}

#[test]
fn extract_if_test() {
    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let deleted = m.extract_if(|v| *v < 4).collect::<Vec<_>>();

    check_links(&m);

    assert_eq!(deleted, &[1, 2, 3]);
    assert_eq!(m.into_iter().collect::<Vec<_>>(), &[4, 5, 6]);
}

#[test]
fn drain_to_empty_test() {
    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let deleted = m.extract_if(|_| true).collect::<Vec<_>>();

    check_links(&m);

    assert_eq!(deleted, &[1, 2, 3, 4, 5, 6]);
    assert_eq!(m.into_iter().collect::<Vec<_>>(), &[]);
}

#[test]
fn test_cursor_move_peek() {
    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let mut cursor = m.cursor_front();
    assert_eq!(cursor.current(), Some(&1));
    assert_eq!(cursor.peek_next(), Some(&2));
    assert_eq!(cursor.peek_prev(), None);
    assert_eq!(cursor.index(), Some(0));
    cursor.move_prev();
    assert_eq!(cursor.current(), None);
    assert_eq!(cursor.peek_next(), Some(&1));
    assert_eq!(cursor.peek_prev(), Some(&6));
    assert_eq!(cursor.index(), None);
    cursor.move_next();
    cursor.move_next();
    assert_eq!(cursor.current(), Some(&2));
    assert_eq!(cursor.peek_next(), Some(&3));
    assert_eq!(cursor.peek_prev(), Some(&1));
    assert_eq!(cursor.index(), Some(1));

    let mut cursor = m.cursor_back();
    assert_eq!(cursor.current(), Some(&6));
    assert_eq!(cursor.peek_next(), None);
    assert_eq!(cursor.peek_prev(), Some(&5));
    assert_eq!(cursor.index(), Some(5));
    cursor.move_next();
    assert_eq!(cursor.current(), None);
    assert_eq!(cursor.peek_next(), Some(&1));
    assert_eq!(cursor.peek_prev(), Some(&6));
    assert_eq!(cursor.index(), None);
    cursor.move_prev();
    cursor.move_prev();
    assert_eq!(cursor.current(), Some(&5));
    assert_eq!(cursor.peek_next(), Some(&6));
    assert_eq!(cursor.peek_prev(), Some(&4));
    assert_eq!(cursor.index(), Some(4));

    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let mut cursor = m.cursor_front_mut();
    assert_eq!(cursor.current(), Some(&mut 1));
    assert_eq!(cursor.peek_next(), Some(&mut 2));
    assert_eq!(cursor.peek_prev(), None);
    assert_eq!(cursor.index(), Some(0));
    cursor.move_prev();
    assert_eq!(cursor.current(), None);
    assert_eq!(cursor.peek_next(), Some(&mut 1));
    assert_eq!(cursor.peek_prev(), Some(&mut 6));
    assert_eq!(cursor.index(), None);
    cursor.move_next();
    cursor.move_next();
    assert_eq!(cursor.current(), Some(&mut 2));
    assert_eq!(cursor.peek_next(), Some(&mut 3));
    assert_eq!(cursor.peek_prev(), Some(&mut 1));
    assert_eq!(cursor.index(), Some(1));
    let mut cursor2 = cursor.as_cursor();
    assert_eq!(cursor2.current(), Some(&2));
    assert_eq!(cursor2.index(), Some(1));
    cursor2.move_next();
    assert_eq!(cursor2.current(), Some(&3));
    assert_eq!(cursor2.index(), Some(2));
    assert_eq!(cursor.current(), Some(&mut 2));
    assert_eq!(cursor.index(), Some(1));

    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let mut cursor = m.cursor_back_mut();
    assert_eq!(cursor.current(), Some(&mut 6));
    assert_eq!(cursor.peek_next(), None);
    assert_eq!(cursor.peek_prev(), Some(&mut 5));
    assert_eq!(cursor.index(), Some(5));
    cursor.move_next();
    assert_eq!(cursor.current(), None);
    assert_eq!(cursor.peek_next(), Some(&mut 1));
    assert_eq!(cursor.peek_prev(), Some(&mut 6));
    assert_eq!(cursor.index(), None);
    cursor.move_prev();
    cursor.move_prev();
    assert_eq!(cursor.current(), Some(&mut 5));
    assert_eq!(cursor.peek_next(), Some(&mut 6));
    assert_eq!(cursor.peek_prev(), Some(&mut 4));
    assert_eq!(cursor.index(), Some(4));
    let mut cursor2 = cursor.as_cursor();
    assert_eq!(cursor2.current(), Some(&5));
    assert_eq!(cursor2.index(), Some(4));
    cursor2.move_prev();
    assert_eq!(cursor2.current(), Some(&4));
    assert_eq!(cursor2.index(), Some(3));
    assert_eq!(cursor.current(), Some(&mut 5));
    assert_eq!(cursor.index(), Some(4));
}

#[test]
fn test_cursor_mut_insert() {
    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let mut cursor = m.cursor_front_mut();
    cursor.insert_before(7);
    cursor.insert_after(8);
    check_links(&m);
    assert_eq!(m.iter().cloned().collect::<Vec<_>>(), &[7, 1, 8, 2, 3, 4, 5, 6]);
    let mut cursor = m.cursor_front_mut();
    cursor.move_prev();
    cursor.insert_before(9);
    cursor.insert_after(10);
    check_links(&m);
    assert_eq!(m.iter().cloned().collect::<Vec<_>>(), &[10, 7, 1, 8, 2, 3, 4, 5, 6, 9]);
    let mut cursor = m.cursor_front_mut();
    cursor.move_prev();
    assert_eq!(cursor.remove_current(), None);
    cursor.move_next();
    cursor.move_next();
    assert_eq!(cursor.remove_current(), Some(7));
    cursor.move_prev();
    cursor.move_prev();
    cursor.move_prev();
    assert_eq!(cursor.remove_current(), Some(9));
    cursor.move_next();
    assert_eq!(cursor.remove_current(), Some(10));
    check_links(&m);
    assert_eq!(m.iter().cloned().collect::<Vec<_>>(), &[1, 8, 2, 3, 4, 5, 6]);
    let mut cursor = m.cursor_front_mut();
    let mut p: LinkedList<u32> = LinkedList::new();
    p.extend(&[100, 101, 102, 103]);
    let mut q: LinkedList<u32> = LinkedList::new();
    q.extend(&[200, 201, 202, 203]);
    cursor.splice_after(p);
    cursor.splice_before(q);
    check_links(&m);
    assert_eq!(m.iter().cloned().collect::<Vec<_>>(), &[
        200, 201, 202, 203, 1, 100, 101, 102, 103, 8, 2, 3, 4, 5, 6
    ]);
    let mut cursor = m.cursor_front_mut();
    cursor.move_prev();
    let tmp = cursor.split_before();
    assert_eq!(m.into_iter().collect::<Vec<_>>(), &[]);
    m = tmp;
    let mut cursor = m.cursor_front_mut();
    cursor.move_next();
    cursor.move_next();
    cursor.move_next();
    cursor.move_next();
    cursor.move_next();
    cursor.move_next();
    let tmp = cursor.split_after();
    assert_eq!(tmp.into_iter().collect::<Vec<_>>(), &[102, 103, 8, 2, 3, 4, 5, 6]);
    check_links(&m);
    assert_eq!(m.iter().cloned().collect::<Vec<_>>(), &[200, 201, 202, 203, 1, 100, 101]);
}

#[test]
fn test_cursor_push_front_back() {
    let mut ll: LinkedList<u32> = LinkedList::new();
    ll.extend(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let mut c = ll.cursor_front_mut();
    assert_eq!(c.current(), Some(&mut 1));
    assert_eq!(c.index(), Some(0));
    c.push_front(0);
    assert_eq!(c.current(), Some(&mut 1));
    assert_eq!(c.peek_prev(), Some(&mut 0));
    assert_eq!(c.index(), Some(1));
    c.push_back(11);
    drop(c);
    let p = ll.cursor_back().front().unwrap();
    assert_eq!(p, &0);
    assert_eq!(ll, (0..12).collect());
    check_links(&ll);
}

#[test]
fn test_cursor_pop_front_back() {
    let mut ll: LinkedList<u32> = LinkedList::new();
    ll.extend(&[1, 2, 3, 4, 5, 6]);
    let mut c = ll.cursor_back_mut();
    assert_eq!(c.pop_front(), Some(1));
    c.move_prev();
    c.move_prev();
    c.move_prev();
    assert_eq!(c.pop_back(), Some(6));
    let c = c.as_cursor();
    assert_eq!(c.front(), Some(&2));
    assert_eq!(c.back(), Some(&5));
    assert_eq!(c.index(), Some(1));
    drop(c);
    assert_eq!(ll, (2..6).collect());
    check_links(&ll);
    let mut c = ll.cursor_back_mut();
    assert_eq!(c.current(), Some(&mut 5));
    assert_eq!(c.index, 3);
    assert_eq!(c.pop_back(), Some(5));
    assert_eq!(c.current(), None);
    assert_eq!(c.index, 3);
    assert_eq!(c.pop_back(), Some(4));
    assert_eq!(c.current(), None);
    assert_eq!(c.index, 2);
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
fn extract_if_empty() {
    let mut list: LinkedList<i32> = LinkedList::new();

    {
        let mut iter = list.extract_if(|_| true);
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
fn extract_if_zst() {
    let mut list: LinkedList<_> = [(), (), (), (), ()].into_iter().collect();
    let initial_len = list.len();
    let mut count = 0;

    {
        let mut iter = list.extract_if(|_| true);
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
fn extract_if_false() {
    let mut list: LinkedList<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

    let initial_len = list.len();
    let mut count = 0;

    {
        let mut iter = list.extract_if(|_| false);
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
fn extract_if_true() {
    let mut list: LinkedList<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

    let initial_len = list.len();
    let mut count = 0;

    {
        let mut iter = list.extract_if(|_| true);
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
fn extract_if_complex() {
    {
        //                [+xxx++++++xxxxx++++x+x++]
        let mut list = [
            1, 2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36, 37,
            39,
        ]
        .into_iter()
        .collect::<LinkedList<_>>();

        let removed = list.extract_if(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(list.len(), 14);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![
            1, 7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39
        ]);
    }

    {
        // [xxx++++++xxxxx++++x+x++]
        let mut list =
            [2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36, 37, 39]
                .into_iter()
                .collect::<LinkedList<_>>();

        let removed = list.extract_if(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(list.len(), 13);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![
            7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39
        ]);
    }

    {
        // [xxx++++++xxxxx++++x+x]
        let mut list =
            [2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36]
                .into_iter()
                .collect::<LinkedList<_>>();

        let removed = list.extract_if(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(list.len(), 11);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![
            7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35
        ]);
    }

    {
        // [xxxxxxxxxx+++++++++++]
        let mut list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            .into_iter()
            .collect::<LinkedList<_>>();

        let removed = list.extract_if(|x| *x % 2 == 0).collect::<Vec<_>>();
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

        let removed = list.extract_if(|x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(list.len(), 10);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn extract_if_drop_panic_leak() {
    let d0 = CrashTestDummy::new(0);
    let d1 = CrashTestDummy::new(1);
    let d2 = CrashTestDummy::new(2);
    let d3 = CrashTestDummy::new(3);
    let d4 = CrashTestDummy::new(4);
    let d5 = CrashTestDummy::new(5);
    let d6 = CrashTestDummy::new(6);
    let d7 = CrashTestDummy::new(7);
    let mut q = LinkedList::new();
    q.push_back(d3.spawn(Panic::Never));
    q.push_back(d4.spawn(Panic::Never));
    q.push_back(d5.spawn(Panic::Never));
    q.push_back(d6.spawn(Panic::Never));
    q.push_back(d7.spawn(Panic::Never));
    q.push_front(d2.spawn(Panic::Never));
    q.push_front(d1.spawn(Panic::InDrop));
    q.push_front(d0.spawn(Panic::Never));

    catch_unwind(AssertUnwindSafe(|| q.extract_if(|_| true).for_each(drop))).unwrap_err();

    assert_eq!(d0.dropped(), 1);
    assert_eq!(d1.dropped(), 1);
    assert_eq!(d2.dropped(), 0);
    assert_eq!(d3.dropped(), 0);
    assert_eq!(d4.dropped(), 0);
    assert_eq!(d5.dropped(), 0);
    assert_eq!(d6.dropped(), 0);
    assert_eq!(d7.dropped(), 0);
    drop(q);
    assert_eq!(d2.dropped(), 1);
    assert_eq!(d3.dropped(), 1);
    assert_eq!(d4.dropped(), 1);
    assert_eq!(d5.dropped(), 1);
    assert_eq!(d6.dropped(), 1);
    assert_eq!(d7.dropped(), 1);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn extract_if_pred_panic_leak() {
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
        q.extract_if(|item| if item.0 >= 2 { panic!() } else { true }).for_each(drop)
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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

#[test]
fn test_allocator() {
    use core::alloc::{AllocError, Allocator, Layout};
    use core::cell::Cell;

    struct A {
        has_allocated: Cell<bool>,
        has_deallocated: Cell<bool>,
    }

    unsafe impl Allocator for A {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            assert!(!self.has_allocated.get());
            self.has_allocated.set(true);

            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            assert!(!self.has_deallocated.get());
            self.has_deallocated.set(true);

            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    let alloc = &A { has_allocated: Cell::new(false), has_deallocated: Cell::new(false) };
    {
        let mut list = LinkedList::new_in(alloc);
        list.push_back(5u32);
        list.remove(0);
    }

    assert!(alloc.has_allocated.get());
    assert!(alloc.has_deallocated.get());
}

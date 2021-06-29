use super::*;

use std::thread;
use std::vec::Vec;

use rand::{thread_rng, RngCore};

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
fn test_fuzz() {
    for _ in 0..25 {
        fuzz_test(3);
        fuzz_test(16);
        #[cfg(not(miri))] // Miri is too slow
        fuzz_test(189);
    }
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

fn fuzz_test(sz: i32) {
    let mut m: LinkedList<_> = LinkedList::new();
    let mut v = vec![];
    for i in 0..sz {
        check_links(&m);
        let r: u8 = thread_rng().next_u32() as u8;
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
fn drain_filter_test() {
    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let deleted = m.drain_filter(|v| *v < 4).collect::<Vec<_>>();

    check_links(&m);

    assert_eq!(deleted, &[1, 2, 3]);
    assert_eq!(m.into_iter().collect::<Vec<_>>(), &[4, 5, 6]);
}

#[test]
fn drain_to_empty_test() {
    let mut m: LinkedList<u32> = LinkedList::new();
    m.extend(&[1, 2, 3, 4, 5, 6]);
    let deleted = m.drain_filter(|_| true).collect::<Vec<_>>();

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
    assert_eq!(
        m.iter().cloned().collect::<Vec<_>>(),
        &[200, 201, 202, 203, 1, 100, 101, 102, 103, 8, 2, 3, 4, 5, 6]
    );
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
    drop(c);
    assert_eq!(ll, (2..6).collect());
    check_links(&ll);
}

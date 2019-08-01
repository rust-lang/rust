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
    // let's make sure it's working properly, since we
    // did some direct changes to private members
    n.push_back(3);
    assert_eq!(n.len(), 1);
    assert_eq!(n.pop_front(), Some(3));
    check_links(&n);
}

#[test]
fn test_insert_prev() {
    let mut m = list_from(&[0, 2, 4, 6, 8]);
    let len = m.len();
    {
        let mut it = m.iter_mut();
        it.insert_next(-2);
        loop {
            match it.next() {
                None => break,
                Some(elt) => {
                    it.insert_next(*elt + 1);
                    match it.peek_next() {
                        Some(x) => assert_eq!(*x, *elt + 2),
                        None => assert_eq!(8, *elt),
                    }
                }
            }
        }
        it.insert_next(0);
        it.insert_next(1);
    }
    check_links(&m);
    assert_eq!(m.len(), 3 + len * 2);
    assert_eq!(m.into_iter().collect::<Vec<_>>(),
                [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]);
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
#[cfg(not(miri))] // Miri does not support threads
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

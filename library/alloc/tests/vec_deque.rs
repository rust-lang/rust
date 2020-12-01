use std::collections::TryReserveError::*;
use std::collections::{vec_deque::Drain, VecDeque};
use std::fmt::Debug;
use std::mem::size_of;
use std::ops::Bound::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::hash;

use Taggy::*;
use Taggypar::*;

#[test]
fn test_simple() {
    let mut d = VecDeque::new();
    assert_eq!(d.len(), 0);
    d.push_front(17);
    d.push_front(42);
    d.push_back(137);
    assert_eq!(d.len(), 3);
    d.push_back(137);
    assert_eq!(d.len(), 4);
    assert_eq!(*d.front().unwrap(), 42);
    assert_eq!(*d.back().unwrap(), 137);
    let mut i = d.pop_front();
    assert_eq!(i, Some(42));
    i = d.pop_back();
    assert_eq!(i, Some(137));
    i = d.pop_back();
    assert_eq!(i, Some(137));
    i = d.pop_back();
    assert_eq!(i, Some(17));
    assert_eq!(d.len(), 0);
    d.push_back(3);
    assert_eq!(d.len(), 1);
    d.push_front(2);
    assert_eq!(d.len(), 2);
    d.push_back(4);
    assert_eq!(d.len(), 3);
    d.push_front(1);
    assert_eq!(d.len(), 4);
    assert_eq!(d[0], 1);
    assert_eq!(d[1], 2);
    assert_eq!(d[2], 3);
    assert_eq!(d[3], 4);
}

fn test_parameterized<T: Clone + PartialEq + Debug>(a: T, b: T, c: T, d: T) {
    let mut deq = VecDeque::new();
    assert_eq!(deq.len(), 0);
    deq.push_front(a.clone());
    deq.push_front(b.clone());
    deq.push_back(c.clone());
    assert_eq!(deq.len(), 3);
    deq.push_back(d.clone());
    assert_eq!(deq.len(), 4);
    assert_eq!((*deq.front().unwrap()).clone(), b.clone());
    assert_eq!((*deq.back().unwrap()).clone(), d.clone());
    assert_eq!(deq.pop_front().unwrap(), b.clone());
    assert_eq!(deq.pop_back().unwrap(), d.clone());
    assert_eq!(deq.pop_back().unwrap(), c.clone());
    assert_eq!(deq.pop_back().unwrap(), a.clone());
    assert_eq!(deq.len(), 0);
    deq.push_back(c.clone());
    assert_eq!(deq.len(), 1);
    deq.push_front(b.clone());
    assert_eq!(deq.len(), 2);
    deq.push_back(d.clone());
    assert_eq!(deq.len(), 3);
    deq.push_front(a.clone());
    assert_eq!(deq.len(), 4);
    assert_eq!(deq[0].clone(), a.clone());
    assert_eq!(deq[1].clone(), b.clone());
    assert_eq!(deq[2].clone(), c.clone());
    assert_eq!(deq[3].clone(), d.clone());
}

#[test]
fn test_push_front_grow() {
    let mut deq = VecDeque::new();
    for i in 0..66 {
        deq.push_front(i);
    }
    assert_eq!(deq.len(), 66);

    for i in 0..66 {
        assert_eq!(deq[i], 65 - i);
    }

    let mut deq = VecDeque::new();
    for i in 0..66 {
        deq.push_back(i);
    }

    for i in 0..66 {
        assert_eq!(deq[i], i);
    }
}

#[test]
fn test_index() {
    let mut deq = VecDeque::new();
    for i in 1..4 {
        deq.push_front(i);
    }
    assert_eq!(deq[1], 2);
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let mut deq = VecDeque::new();
    for i in 1..4 {
        deq.push_front(i);
    }
    deq[3];
}

#[test]
#[should_panic]
fn test_range_start_overflow() {
    let deq = VecDeque::from(vec![1, 2, 3]);
    deq.range((Included(0), Included(usize::MAX)));
}

#[test]
#[should_panic]
fn test_range_end_overflow() {
    let deq = VecDeque::from(vec![1, 2, 3]);
    deq.range((Excluded(usize::MAX), Included(0)));
}

#[derive(Clone, PartialEq, Debug)]
enum Taggy {
    One(i32),
    Two(i32, i32),
    Three(i32, i32, i32),
}

#[derive(Clone, PartialEq, Debug)]
enum Taggypar<T> {
    Onepar(T),
    Twopar(T, T),
    Threepar(T, T, T),
}

#[derive(Clone, PartialEq, Debug)]
struct RecCy {
    x: i32,
    y: i32,
    t: Taggy,
}

#[test]
fn test_param_int() {
    test_parameterized::<i32>(5, 72, 64, 175);
}

#[test]
fn test_param_taggy() {
    test_parameterized::<Taggy>(One(1), Two(1, 2), Three(1, 2, 3), Two(17, 42));
}

#[test]
fn test_param_taggypar() {
    test_parameterized::<Taggypar<i32>>(
        Onepar::<i32>(1),
        Twopar::<i32>(1, 2),
        Threepar::<i32>(1, 2, 3),
        Twopar::<i32>(17, 42),
    );
}

#[test]
fn test_param_reccy() {
    let reccy1 = RecCy { x: 1, y: 2, t: One(1) };
    let reccy2 = RecCy { x: 345, y: 2, t: Two(1, 2) };
    let reccy3 = RecCy { x: 1, y: 777, t: Three(1, 2, 3) };
    let reccy4 = RecCy { x: 19, y: 252, t: Two(17, 42) };
    test_parameterized::<RecCy>(reccy1, reccy2, reccy3, reccy4);
}

#[test]
fn test_with_capacity() {
    let mut d = VecDeque::with_capacity(0);
    d.push_back(1);
    assert_eq!(d.len(), 1);
    let mut d = VecDeque::with_capacity(50);
    d.push_back(1);
    assert_eq!(d.len(), 1);
}

#[test]
fn test_with_capacity_non_power_two() {
    let mut d3 = VecDeque::with_capacity(3);
    d3.push_back(1);

    // X = None, | = lo
    // [|1, X, X]
    assert_eq!(d3.pop_front(), Some(1));
    // [X, |X, X]
    assert_eq!(d3.front(), None);

    // [X, |3, X]
    d3.push_back(3);
    // [X, |3, 6]
    d3.push_back(6);
    // [X, X, |6]
    assert_eq!(d3.pop_front(), Some(3));

    // Pushing the lo past half way point to trigger
    // the 'B' scenario for growth
    // [9, X, |6]
    d3.push_back(9);
    // [9, 12, |6]
    d3.push_back(12);

    d3.push_back(15);
    // There used to be a bug here about how the
    // VecDeque made growth assumptions about the
    // underlying Vec which didn't hold and lead
    // to corruption.
    // (Vec grows to next power of two)
    // good- [9, 12, 15, X, X, X, X, |6]
    // bug-  [15, 12, X, X, X, |6, X, X]
    assert_eq!(d3.pop_front(), Some(6));

    // Which leads us to the following state which
    // would be a failure case.
    // bug-  [15, 12, X, X, X, X, |X, X]
    assert_eq!(d3.front(), Some(&9));
}

#[test]
fn test_reserve_exact() {
    let mut d = VecDeque::new();
    d.push_back(0);
    d.reserve_exact(50);
    assert!(d.capacity() >= 51);
}

#[test]
fn test_reserve() {
    let mut d = VecDeque::new();
    d.push_back(0);
    d.reserve(50);
    assert!(d.capacity() >= 51);
}

#[test]
fn test_swap() {
    let mut d: VecDeque<_> = (0..5).collect();
    d.pop_front();
    d.swap(0, 3);
    assert_eq!(d.iter().cloned().collect::<Vec<_>>(), [4, 2, 3, 1]);
}

#[test]
fn test_iter() {
    let mut d = VecDeque::new();
    assert_eq!(d.iter().next(), None);
    assert_eq!(d.iter().size_hint(), (0, Some(0)));

    for i in 0..5 {
        d.push_back(i);
    }
    {
        let b: &[_] = &[&0, &1, &2, &3, &4];
        assert_eq!(d.iter().collect::<Vec<_>>(), b);
    }

    for i in 6..9 {
        d.push_front(i);
    }
    {
        let b: &[_] = &[&8, &7, &6, &0, &1, &2, &3, &4];
        assert_eq!(d.iter().collect::<Vec<_>>(), b);
    }

    let mut it = d.iter();
    let mut len = d.len();
    loop {
        match it.next() {
            None => break,
            _ => {
                len -= 1;
                assert_eq!(it.size_hint(), (len, Some(len)))
            }
        }
    }
}

#[test]
fn test_rev_iter() {
    let mut d = VecDeque::new();
    assert_eq!(d.iter().rev().next(), None);

    for i in 0..5 {
        d.push_back(i);
    }
    {
        let b: &[_] = &[&4, &3, &2, &1, &0];
        assert_eq!(d.iter().rev().collect::<Vec<_>>(), b);
    }

    for i in 6..9 {
        d.push_front(i);
    }
    let b: &[_] = &[&4, &3, &2, &1, &0, &6, &7, &8];
    assert_eq!(d.iter().rev().collect::<Vec<_>>(), b);
}

#[test]
fn test_mut_rev_iter_wrap() {
    let mut d = VecDeque::with_capacity(3);
    assert!(d.iter_mut().rev().next().is_none());

    d.push_back(1);
    d.push_back(2);
    d.push_back(3);
    assert_eq!(d.pop_front(), Some(1));
    d.push_back(4);

    assert_eq!(d.iter_mut().rev().map(|x| *x).collect::<Vec<_>>(), vec![4, 3, 2]);
}

#[test]
fn test_mut_iter() {
    let mut d = VecDeque::new();
    assert!(d.iter_mut().next().is_none());

    for i in 0..3 {
        d.push_front(i);
    }

    for (i, elt) in d.iter_mut().enumerate() {
        assert_eq!(*elt, 2 - i);
        *elt = i;
    }

    {
        let mut it = d.iter_mut();
        assert_eq!(*it.next().unwrap(), 0);
        assert_eq!(*it.next().unwrap(), 1);
        assert_eq!(*it.next().unwrap(), 2);
        assert!(it.next().is_none());
    }
}

#[test]
fn test_mut_rev_iter() {
    let mut d = VecDeque::new();
    assert!(d.iter_mut().rev().next().is_none());

    for i in 0..3 {
        d.push_front(i);
    }

    for (i, elt) in d.iter_mut().rev().enumerate() {
        assert_eq!(*elt, i);
        *elt = i;
    }

    {
        let mut it = d.iter_mut().rev();
        assert_eq!(*it.next().unwrap(), 0);
        assert_eq!(*it.next().unwrap(), 1);
        assert_eq!(*it.next().unwrap(), 2);
        assert!(it.next().is_none());
    }
}

#[test]
fn test_into_iter() {
    // Empty iter
    {
        let d: VecDeque<i32> = VecDeque::new();
        let mut iter = d.into_iter();

        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    // simple iter
    {
        let mut d = VecDeque::new();
        for i in 0..5 {
            d.push_back(i);
        }

        let b = vec![0, 1, 2, 3, 4];
        assert_eq!(d.into_iter().collect::<Vec<_>>(), b);
    }

    // wrapped iter
    {
        let mut d = VecDeque::new();
        for i in 0..5 {
            d.push_back(i);
        }
        for i in 6..9 {
            d.push_front(i);
        }

        let b = vec![8, 7, 6, 0, 1, 2, 3, 4];
        assert_eq!(d.into_iter().collect::<Vec<_>>(), b);
    }

    // partially used
    {
        let mut d = VecDeque::new();
        for i in 0..5 {
            d.push_back(i);
        }
        for i in 6..9 {
            d.push_front(i);
        }

        let mut it = d.into_iter();
        assert_eq!(it.size_hint(), (8, Some(8)));
        assert_eq!(it.next(), Some(8));
        assert_eq!(it.size_hint(), (7, Some(7)));
        assert_eq!(it.next_back(), Some(4));
        assert_eq!(it.size_hint(), (6, Some(6)));
        assert_eq!(it.next(), Some(7));
        assert_eq!(it.size_hint(), (5, Some(5)));
    }
}

#[test]
fn test_drain() {
    // Empty iter
    {
        let mut d: VecDeque<i32> = VecDeque::new();

        {
            let mut iter = d.drain(..);

            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
            assert_eq!(iter.size_hint(), (0, Some(0)));
        }

        assert!(d.is_empty());
    }

    // simple iter
    {
        let mut d = VecDeque::new();
        for i in 0..5 {
            d.push_back(i);
        }

        assert_eq!(d.drain(..).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        assert!(d.is_empty());
    }

    // wrapped iter
    {
        let mut d = VecDeque::new();
        for i in 0..5 {
            d.push_back(i);
        }
        for i in 6..9 {
            d.push_front(i);
        }

        assert_eq!(d.drain(..).collect::<Vec<_>>(), [8, 7, 6, 0, 1, 2, 3, 4]);
        assert!(d.is_empty());
    }

    // partially used
    {
        let mut d: VecDeque<_> = VecDeque::new();
        for i in 0..5 {
            d.push_back(i);
        }
        for i in 6..9 {
            d.push_front(i);
        }

        {
            let mut it = d.drain(..);
            assert_eq!(it.size_hint(), (8, Some(8)));
            assert_eq!(it.next(), Some(8));
            assert_eq!(it.size_hint(), (7, Some(7)));
            assert_eq!(it.next_back(), Some(4));
            assert_eq!(it.size_hint(), (6, Some(6)));
            assert_eq!(it.next(), Some(7));
            assert_eq!(it.size_hint(), (5, Some(5)));
        }
        assert!(d.is_empty());
    }
}

#[test]
fn test_from_iter() {
    let v = vec![1, 2, 3, 4, 5, 6, 7];
    let deq: VecDeque<_> = v.iter().cloned().collect();
    let u: Vec<_> = deq.iter().cloned().collect();
    assert_eq!(u, v);

    let seq = (0..).step_by(2).take(256);
    let deq: VecDeque<_> = seq.collect();
    for (i, &x) in deq.iter().enumerate() {
        assert_eq!(2 * i, x);
    }
    assert_eq!(deq.len(), 256);
}

#[test]
fn test_clone() {
    let mut d = VecDeque::new();
    d.push_front(17);
    d.push_front(42);
    d.push_back(137);
    d.push_back(137);
    assert_eq!(d.len(), 4);
    let mut e = d.clone();
    assert_eq!(e.len(), 4);
    while !d.is_empty() {
        assert_eq!(d.pop_back(), e.pop_back());
    }
    assert_eq!(d.len(), 0);
    assert_eq!(e.len(), 0);
}

#[test]
fn test_eq() {
    let mut d = VecDeque::new();
    assert!(d == VecDeque::with_capacity(0));
    d.push_front(137);
    d.push_front(17);
    d.push_front(42);
    d.push_back(137);
    let mut e = VecDeque::with_capacity(0);
    e.push_back(42);
    e.push_back(17);
    e.push_back(137);
    e.push_back(137);
    assert!(&e == &d);
    e.pop_back();
    e.push_back(0);
    assert!(e != d);
    e.clear();
    assert!(e == VecDeque::new());
}

#[test]
fn test_partial_eq_array() {
    let d = VecDeque::<char>::new();
    assert!(d == []);

    let mut d = VecDeque::new();
    d.push_front('a');
    assert!(d == ['a']);

    let mut d = VecDeque::new();
    d.push_back('a');
    assert!(d == ['a']);

    let mut d = VecDeque::new();
    d.push_back('a');
    d.push_back('b');
    assert!(d == ['a', 'b']);
}

#[test]
fn test_hash() {
    let mut x = VecDeque::new();
    let mut y = VecDeque::new();

    x.push_back(1);
    x.push_back(2);
    x.push_back(3);

    y.push_back(0);
    y.push_back(1);
    y.pop_front();
    y.push_back(2);
    y.push_back(3);

    assert!(hash(&x) == hash(&y));
}

#[test]
fn test_hash_after_rotation() {
    // test that two deques hash equal even if elements are laid out differently
    let len = 28;
    let mut ring: VecDeque<i32> = (0..len as i32).collect();
    let orig = ring.clone();
    for _ in 0..ring.capacity() {
        // shift values 1 step to the right by pop, sub one, push
        ring.pop_front();
        for elt in &mut ring {
            *elt -= 1;
        }
        ring.push_back(len - 1);
        assert_eq!(hash(&orig), hash(&ring));
        assert_eq!(orig, ring);
        assert_eq!(ring, orig);
    }
}

#[test]
fn test_eq_after_rotation() {
    // test that two deques are equal even if elements are laid out differently
    let len = 28;
    let mut ring: VecDeque<i32> = (0..len as i32).collect();
    let mut shifted = ring.clone();
    for _ in 0..10 {
        // shift values 1 step to the right by pop, sub one, push
        ring.pop_front();
        for elt in &mut ring {
            *elt -= 1;
        }
        ring.push_back(len - 1);
    }

    // try every shift
    for _ in 0..shifted.capacity() {
        shifted.pop_front();
        for elt in &mut shifted {
            *elt -= 1;
        }
        shifted.push_back(len - 1);
        assert_eq!(shifted, ring);
        assert_eq!(ring, shifted);
    }
}

#[test]
fn test_ord() {
    let x = VecDeque::new();
    let mut y = VecDeque::new();
    y.push_back(1);
    y.push_back(2);
    y.push_back(3);
    assert!(x < y);
    assert!(y > x);
    assert!(x <= x);
    assert!(x >= x);
}

#[test]
fn test_show() {
    let ringbuf: VecDeque<_> = (0..10).collect();
    assert_eq!(format!("{:?}", ringbuf), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

    let ringbuf: VecDeque<_> = vec!["just", "one", "test", "more"].iter().cloned().collect();
    assert_eq!(format!("{:?}", ringbuf), "[\"just\", \"one\", \"test\", \"more\"]");
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

    let mut ring = VecDeque::new();
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

    let mut ring = VecDeque::new();
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

    let mut ring = VecDeque::new();
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

    let mut q = VecDeque::new();
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
fn test_reserve_grow() {
    // test growth path A
    // [T o o H] -> [T o o H . . . . ]
    let mut ring = VecDeque::with_capacity(4);
    for i in 0..3 {
        ring.push_back(i);
    }
    ring.reserve(7);
    for i in 0..3 {
        assert_eq!(ring.pop_front(), Some(i));
    }

    // test growth path B
    // [H T o o] -> [. T o o H . . . ]
    let mut ring = VecDeque::with_capacity(4);
    for i in 0..1 {
        ring.push_back(i);
        assert_eq!(ring.pop_front(), Some(i));
    }
    for i in 0..3 {
        ring.push_back(i);
    }
    ring.reserve(7);
    for i in 0..3 {
        assert_eq!(ring.pop_front(), Some(i));
    }

    // test growth path C
    // [o o H T] -> [o o H . . . . T ]
    let mut ring = VecDeque::with_capacity(4);
    for i in 0..3 {
        ring.push_back(i);
        assert_eq!(ring.pop_front(), Some(i));
    }
    for i in 0..3 {
        ring.push_back(i);
    }
    ring.reserve(7);
    for i in 0..3 {
        assert_eq!(ring.pop_front(), Some(i));
    }
}

#[test]
fn test_get() {
    let mut ring = VecDeque::new();
    ring.push_back(0);
    assert_eq!(ring.get(0), Some(&0));
    assert_eq!(ring.get(1), None);

    ring.push_back(1);
    assert_eq!(ring.get(0), Some(&0));
    assert_eq!(ring.get(1), Some(&1));
    assert_eq!(ring.get(2), None);

    ring.push_back(2);
    assert_eq!(ring.get(0), Some(&0));
    assert_eq!(ring.get(1), Some(&1));
    assert_eq!(ring.get(2), Some(&2));
    assert_eq!(ring.get(3), None);

    assert_eq!(ring.pop_front(), Some(0));
    assert_eq!(ring.get(0), Some(&1));
    assert_eq!(ring.get(1), Some(&2));
    assert_eq!(ring.get(2), None);

    assert_eq!(ring.pop_front(), Some(1));
    assert_eq!(ring.get(0), Some(&2));
    assert_eq!(ring.get(1), None);

    assert_eq!(ring.pop_front(), Some(2));
    assert_eq!(ring.get(0), None);
    assert_eq!(ring.get(1), None);
}

#[test]
fn test_get_mut() {
    let mut ring = VecDeque::new();
    for i in 0..3 {
        ring.push_back(i);
    }

    match ring.get_mut(1) {
        Some(x) => *x = -1,
        None => (),
    };

    assert_eq!(ring.get_mut(0), Some(&mut 0));
    assert_eq!(ring.get_mut(1), Some(&mut -1));
    assert_eq!(ring.get_mut(2), Some(&mut 2));
    assert_eq!(ring.get_mut(3), None);

    assert_eq!(ring.pop_front(), Some(0));
    assert_eq!(ring.get_mut(0), Some(&mut -1));
    assert_eq!(ring.get_mut(1), Some(&mut 2));
    assert_eq!(ring.get_mut(2), None);
}

#[test]
fn test_front() {
    let mut ring = VecDeque::new();
    ring.push_back(10);
    ring.push_back(20);
    assert_eq!(ring.front(), Some(&10));
    ring.pop_front();
    assert_eq!(ring.front(), Some(&20));
    ring.pop_front();
    assert_eq!(ring.front(), None);
}

#[test]
fn test_as_slices() {
    let mut ring: VecDeque<i32> = VecDeque::with_capacity(127);
    let cap = ring.capacity() as i32;
    let first = cap / 2;
    let last = cap - first;
    for i in 0..first {
        ring.push_back(i);

        let (left, right) = ring.as_slices();
        let expected: Vec<_> = (0..=i).collect();
        assert_eq!(left, &expected[..]);
        assert_eq!(right, []);
    }

    for j in -last..0 {
        ring.push_front(j);
        let (left, right) = ring.as_slices();
        let expected_left: Vec<_> = (-last..=j).rev().collect();
        let expected_right: Vec<_> = (0..first).collect();
        assert_eq!(left, &expected_left[..]);
        assert_eq!(right, &expected_right[..]);
    }

    assert_eq!(ring.len() as i32, cap);
    assert_eq!(ring.capacity() as i32, cap);
}

#[test]
fn test_as_mut_slices() {
    let mut ring: VecDeque<i32> = VecDeque::with_capacity(127);
    let cap = ring.capacity() as i32;
    let first = cap / 2;
    let last = cap - first;
    for i in 0..first {
        ring.push_back(i);

        let (left, right) = ring.as_mut_slices();
        let expected: Vec<_> = (0..=i).collect();
        assert_eq!(left, &expected[..]);
        assert_eq!(right, []);
    }

    for j in -last..0 {
        ring.push_front(j);
        let (left, right) = ring.as_mut_slices();
        let expected_left: Vec<_> = (-last..=j).rev().collect();
        let expected_right: Vec<_> = (0..first).collect();
        assert_eq!(left, &expected_left[..]);
        assert_eq!(right, &expected_right[..]);
    }

    assert_eq!(ring.len() as i32, cap);
    assert_eq!(ring.capacity() as i32, cap);
}

#[test]
fn test_append() {
    let mut a: VecDeque<_> = vec![1, 2, 3].into_iter().collect();
    let mut b: VecDeque<_> = vec![4, 5, 6].into_iter().collect();

    // normal append
    a.append(&mut b);
    assert_eq!(a.iter().cloned().collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(b.iter().cloned().collect::<Vec<_>>(), []);

    // append nothing to something
    a.append(&mut b);
    assert_eq!(a.iter().cloned().collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(b.iter().cloned().collect::<Vec<_>>(), []);

    // append something to nothing
    b.append(&mut a);
    assert_eq!(b.iter().cloned().collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(a.iter().cloned().collect::<Vec<_>>(), []);
}

#[test]
fn test_append_permutations() {
    fn construct_vec_deque(
        push_back: usize,
        pop_back: usize,
        push_front: usize,
        pop_front: usize,
    ) -> VecDeque<usize> {
        let mut out = VecDeque::new();
        for a in 0..push_back {
            out.push_back(a);
        }
        for b in 0..push_front {
            out.push_front(push_back + b);
        }
        for _ in 0..pop_back {
            out.pop_back();
        }
        for _ in 0..pop_front {
            out.pop_front();
        }
        out
    }

    // Miri is too slow
    let max = if cfg!(miri) { 3 } else { 5 };

    // Many different permutations of both the `VecDeque` getting appended to
    // and the one getting appended are generated to check `append`.
    // This ensures all 6 code paths of `append` are tested.
    for src_push_back in 0..max {
        for src_push_front in 0..max {
            // doesn't pop more values than are pushed
            for src_pop_back in 0..(src_push_back + src_push_front) {
                for src_pop_front in 0..(src_push_back + src_push_front - src_pop_back) {
                    let src = construct_vec_deque(
                        src_push_back,
                        src_pop_back,
                        src_push_front,
                        src_pop_front,
                    );

                    for dst_push_back in 0..max {
                        for dst_push_front in 0..max {
                            for dst_pop_back in 0..(dst_push_back + dst_push_front) {
                                for dst_pop_front in
                                    0..(dst_push_back + dst_push_front - dst_pop_back)
                                {
                                    let mut dst = construct_vec_deque(
                                        dst_push_back,
                                        dst_pop_back,
                                        dst_push_front,
                                        dst_pop_front,
                                    );
                                    let mut src = src.clone();

                                    // Assert that appending `src` to `dst` gives the same order
                                    // of values as iterating over both in sequence.
                                    let correct = dst
                                        .iter()
                                        .chain(src.iter())
                                        .cloned()
                                        .collect::<Vec<usize>>();
                                    dst.append(&mut src);
                                    assert_eq!(dst, correct);
                                    assert!(src.is_empty());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

struct DropCounter<'a> {
    count: &'a mut u32,
}

impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
        *self.count += 1;
    }
}

#[test]
fn test_append_double_drop() {
    let (mut count_a, mut count_b) = (0, 0);
    {
        let mut a = VecDeque::new();
        let mut b = VecDeque::new();
        a.push_back(DropCounter { count: &mut count_a });
        b.push_back(DropCounter { count: &mut count_b });

        a.append(&mut b);
    }
    assert_eq!(count_a, 1);
    assert_eq!(count_b, 1);
}

#[test]
fn test_retain() {
    let mut buf = VecDeque::new();
    buf.extend(1..5);
    buf.retain(|&x| x % 2 == 0);
    let v: Vec<_> = buf.into_iter().collect();
    assert_eq!(&v[..], &[2, 4]);
}

#[test]
fn test_extend_ref() {
    let mut v = VecDeque::new();
    v.push_back(1);
    v.extend(&[2, 3, 4]);

    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 3);
    assert_eq!(v[3], 4);

    let mut w = VecDeque::new();
    w.push_back(5);
    w.push_back(6);
    v.extend(&w);

    assert_eq!(v.len(), 6);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 3);
    assert_eq!(v[3], 4);
    assert_eq!(v[4], 5);
    assert_eq!(v[5], 6);
}

#[test]
fn test_contains() {
    let mut v = VecDeque::new();
    v.extend(&[2, 3, 4]);

    assert!(v.contains(&3));
    assert!(!v.contains(&1));

    v.clear();

    assert!(!v.contains(&3));
}

#[allow(dead_code)]
fn assert_covariance() {
    fn drain<'new>(d: Drain<'static, &'static str>) -> Drain<'new, &'new str> {
        d
    }
}

#[test]
fn test_is_empty() {
    let mut v = VecDeque::<i32>::new();
    assert!(v.is_empty());
    assert!(v.iter().is_empty());
    assert!(v.iter_mut().is_empty());
    v.extend(&[2, 3, 4]);
    assert!(!v.is_empty());
    assert!(!v.iter().is_empty());
    assert!(!v.iter_mut().is_empty());
    while let Some(_) = v.pop_front() {
        assert_eq!(v.is_empty(), v.len() == 0);
        assert_eq!(v.iter().is_empty(), v.iter().len() == 0);
        assert_eq!(v.iter_mut().is_empty(), v.iter_mut().len() == 0);
    }
    assert!(v.is_empty());
    assert!(v.iter().is_empty());
    assert!(v.iter_mut().is_empty());
    assert!(v.into_iter().is_empty());
}

#[test]
fn test_reserve_exact_2() {
    // This is all the same as test_reserve

    let mut v = VecDeque::new();

    v.reserve_exact(2);
    assert!(v.capacity() >= 2);

    for i in 0..16 {
        v.push_back(i);
    }

    assert!(v.capacity() >= 16);
    v.reserve_exact(16);
    assert!(v.capacity() >= 32);

    v.push_back(16);

    v.reserve_exact(16);
    assert!(v.capacity() >= 48)
}

#[test]
#[cfg_attr(miri, ignore)] // Miri does not support signalling OOM
#[cfg_attr(target_os = "android", ignore)] // Android used in CI has a broken dlmalloc
fn test_try_reserve() {
    // These are the interesting cases:
    // * exactly isize::MAX should never trigger a CapacityOverflow (can be OOM)
    // * > isize::MAX should always fail
    //    * On 16/32-bit should CapacityOverflow
    //    * On 64-bit should OOM
    // * overflow may trigger when adding `len` to `cap` (in number of elements)
    // * overflow may trigger when multiplying `new_cap` by size_of::<T> (to get bytes)

    const MAX_CAP: usize = (isize::MAX as usize + 1) / 2 - 1;
    const MAX_USIZE: usize = usize::MAX;

    // On 16/32-bit, we check that allocations don't exceed isize::MAX,
    // on 64-bit, we assume the OS will give an OOM for such a ridiculous size.
    // Any platform that succeeds for these requests is technically broken with
    // ptr::offset because LLVM is the worst.
    let guards_against_isize = size_of::<usize>() < 8;

    {
        // Note: basic stuff is checked by test_reserve
        let mut empty_bytes: VecDeque<u8> = VecDeque::new();

        // Check isize::MAX doesn't count as an overflow
        if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_CAP) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        // Play it again, frank! (just to be sure)
        if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_CAP) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        if guards_against_isize {
            // Check isize::MAX + 1 does count as overflow
            if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_CAP + 1) {
            } else {
                panic!("isize::MAX + 1 should trigger an overflow!")
            }

            // Check usize::MAX does count as overflow
            if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_USIZE) {
            } else {
                panic!("usize::MAX should trigger an overflow!")
            }
        } else {
            // Check isize::MAX is an OOM
            // VecDeque starts with capacity 7, always adds 1 to the capacity
            // and also rounds the number to next power of 2 so this is the
            // furthest we can go without triggering CapacityOverflow
            if let Err(AllocError { .. }) = empty_bytes.try_reserve(MAX_CAP) {
            } else {
                panic!("isize::MAX + 1 should trigger an OOM!")
            }
        }
    }

    {
        // Same basic idea, but with non-zero len
        let mut ten_bytes: VecDeque<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

        if let Err(CapacityOverflow) = ten_bytes.try_reserve(MAX_CAP - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = ten_bytes.try_reserve(MAX_CAP - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if guards_against_isize {
            if let Err(CapacityOverflow) = ten_bytes.try_reserve(MAX_CAP - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an overflow!");
            }
        } else {
            if let Err(AllocError { .. }) = ten_bytes.try_reserve(MAX_CAP - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an OOM!")
            }
        }
        // Should always overflow in the add-to-len
        if let Err(CapacityOverflow) = ten_bytes.try_reserve(MAX_USIZE) {
        } else {
            panic!("usize::MAX should trigger an overflow!")
        }
    }

    {
        // Same basic idea, but with interesting type size
        let mut ten_u32s: VecDeque<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

        if let Err(CapacityOverflow) = ten_u32s.try_reserve(MAX_CAP / 4 - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = ten_u32s.try_reserve(MAX_CAP / 4 - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if guards_against_isize {
            if let Err(CapacityOverflow) = ten_u32s.try_reserve(MAX_CAP / 4 - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an overflow!");
            }
        } else {
            if let Err(AllocError { .. }) = ten_u32s.try_reserve(MAX_CAP / 4 - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an OOM!")
            }
        }
        // Should fail in the mul-by-size
        if let Err(CapacityOverflow) = ten_u32s.try_reserve(MAX_USIZE - 20) {
        } else {
            panic!("usize::MAX should trigger an overflow!");
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Miri does not support signalling OOM
#[cfg_attr(target_os = "android", ignore)] // Android used in CI has a broken dlmalloc
fn test_try_reserve_exact() {
    // This is exactly the same as test_try_reserve with the method changed.
    // See that test for comments.

    const MAX_CAP: usize = (isize::MAX as usize + 1) / 2 - 1;
    const MAX_USIZE: usize = usize::MAX;

    let guards_against_isize = size_of::<usize>() < 8;

    {
        let mut empty_bytes: VecDeque<u8> = VecDeque::new();

        if let Err(CapacityOverflow) = empty_bytes.try_reserve_exact(MAX_CAP) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = empty_bytes.try_reserve_exact(MAX_CAP) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        if guards_against_isize {
            if let Err(CapacityOverflow) = empty_bytes.try_reserve_exact(MAX_CAP + 1) {
            } else {
                panic!("isize::MAX + 1 should trigger an overflow!")
            }

            if let Err(CapacityOverflow) = empty_bytes.try_reserve_exact(MAX_USIZE) {
            } else {
                panic!("usize::MAX should trigger an overflow!")
            }
        } else {
            // Check isize::MAX is an OOM
            // VecDeque starts with capacity 7, always adds 1 to the capacity
            // and also rounds the number to next power of 2 so this is the
            // furthest we can go without triggering CapacityOverflow
            if let Err(AllocError { .. }) = empty_bytes.try_reserve_exact(MAX_CAP) {
            } else {
                panic!("isize::MAX + 1 should trigger an OOM!")
            }
        }
    }

    {
        let mut ten_bytes: VecDeque<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

        if let Err(CapacityOverflow) = ten_bytes.try_reserve_exact(MAX_CAP - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = ten_bytes.try_reserve_exact(MAX_CAP - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if guards_against_isize {
            if let Err(CapacityOverflow) = ten_bytes.try_reserve_exact(MAX_CAP - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an overflow!");
            }
        } else {
            if let Err(AllocError { .. }) = ten_bytes.try_reserve_exact(MAX_CAP - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an OOM!")
            }
        }
        if let Err(CapacityOverflow) = ten_bytes.try_reserve_exact(MAX_USIZE) {
        } else {
            panic!("usize::MAX should trigger an overflow!")
        }
    }

    {
        let mut ten_u32s: VecDeque<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();

        if let Err(CapacityOverflow) = ten_u32s.try_reserve_exact(MAX_CAP / 4 - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = ten_u32s.try_reserve_exact(MAX_CAP / 4 - 10) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if guards_against_isize {
            if let Err(CapacityOverflow) = ten_u32s.try_reserve_exact(MAX_CAP / 4 - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an overflow!");
            }
        } else {
            if let Err(AllocError { .. }) = ten_u32s.try_reserve_exact(MAX_CAP / 4 - 9) {
            } else {
                panic!("isize::MAX + 1 should trigger an OOM!")
            }
        }
        if let Err(CapacityOverflow) = ten_u32s.try_reserve_exact(MAX_USIZE - 20) {
        } else {
            panic!("usize::MAX should trigger an overflow!")
        }
    }
}

#[test]
fn test_rotate_nop() {
    let mut v: VecDeque<_> = (0..10).collect();
    assert_unchanged(&v);

    v.rotate_left(0);
    assert_unchanged(&v);

    v.rotate_left(10);
    assert_unchanged(&v);

    v.rotate_right(0);
    assert_unchanged(&v);

    v.rotate_right(10);
    assert_unchanged(&v);

    v.rotate_left(3);
    v.rotate_right(3);
    assert_unchanged(&v);

    v.rotate_right(3);
    v.rotate_left(3);
    assert_unchanged(&v);

    v.rotate_left(6);
    v.rotate_right(6);
    assert_unchanged(&v);

    v.rotate_right(6);
    v.rotate_left(6);
    assert_unchanged(&v);

    v.rotate_left(3);
    v.rotate_left(7);
    assert_unchanged(&v);

    v.rotate_right(4);
    v.rotate_right(6);
    assert_unchanged(&v);

    v.rotate_left(1);
    v.rotate_left(2);
    v.rotate_left(3);
    v.rotate_left(4);
    assert_unchanged(&v);

    v.rotate_right(1);
    v.rotate_right(2);
    v.rotate_right(3);
    v.rotate_right(4);
    assert_unchanged(&v);

    fn assert_unchanged(v: &VecDeque<i32>) {
        assert_eq!(v, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}

#[test]
fn test_rotate_left_parts() {
    let mut v: VecDeque<_> = (1..=7).collect();
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[3, 4, 5, 6, 7, 1][..], &[2][..]));
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[5, 6, 7, 1][..], &[2, 3, 4][..]));
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[7, 1][..], &[2, 3, 4, 5, 6][..]));
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[2, 3, 4, 5, 6, 7, 1][..], &[][..]));
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[4, 5, 6, 7, 1, 2][..], &[3][..]));
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[6, 7, 1, 2][..], &[3, 4, 5][..]));
    v.rotate_left(2);
    assert_eq!(v.as_slices(), (&[1, 2][..], &[3, 4, 5, 6, 7][..]));
}

#[test]
fn test_rotate_right_parts() {
    let mut v: VecDeque<_> = (1..=7).collect();
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[6, 7][..], &[1, 2, 3, 4, 5][..]));
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[4, 5, 6, 7][..], &[1, 2, 3][..]));
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[2, 3, 4, 5, 6, 7][..], &[1][..]));
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[7, 1, 2, 3, 4, 5, 6][..], &[][..]));
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[5, 6][..], &[7, 1, 2, 3, 4][..]));
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[3, 4, 5, 6][..], &[7, 1, 2][..]));
    v.rotate_right(2);
    assert_eq!(v.as_slices(), (&[1, 2, 3, 4, 5, 6][..], &[7][..]));
}

#[test]
fn test_rotate_left_random() {
    let shifts = [
        6, 1, 0, 11, 12, 1, 11, 7, 9, 3, 6, 1, 4, 0, 5, 1, 3, 1, 12, 8, 3, 1, 11, 11, 9, 4, 12, 3,
        12, 9, 11, 1, 7, 9, 7, 2,
    ];
    let n = 12;
    let mut v: VecDeque<_> = (0..n).collect();
    let mut total_shift = 0;
    for shift in shifts.iter().cloned() {
        v.rotate_left(shift);
        total_shift += shift;
        for i in 0..n {
            assert_eq!(v[i], (i + total_shift) % n);
        }
    }
}

#[test]
fn test_rotate_right_random() {
    let shifts = [
        6, 1, 0, 11, 12, 1, 11, 7, 9, 3, 6, 1, 4, 0, 5, 1, 3, 1, 12, 8, 3, 1, 11, 11, 9, 4, 12, 3,
        12, 9, 11, 1, 7, 9, 7, 2,
    ];
    let n = 12;
    let mut v: VecDeque<_> = (0..n).collect();
    let mut total_shift = 0;
    for shift in shifts.iter().cloned() {
        v.rotate_right(shift);
        total_shift += shift;
        for i in 0..n {
            assert_eq!(v[(i + total_shift) % n], i);
        }
    }
}

#[test]
fn test_try_fold_empty() {
    assert_eq!(Some(0), VecDeque::<u32>::new().iter().try_fold(0, |_, _| None));
}

#[test]
fn test_try_fold_none() {
    let v: VecDeque<u32> = (0..12).collect();
    assert_eq!(None, v.into_iter().try_fold(0, |a, b| if b < 11 { Some(a + b) } else { None }));
}

#[test]
fn test_try_fold_ok() {
    let v: VecDeque<u32> = (0..12).collect();
    assert_eq!(Ok::<_, ()>(66), v.into_iter().try_fold(0, |a, b| Ok(a + b)));
}

#[test]
fn test_try_fold_unit() {
    let v: VecDeque<()> = std::iter::repeat(()).take(42).collect();
    assert_eq!(Some(()), v.into_iter().try_fold((), |(), ()| Some(())));
}

#[test]
fn test_try_fold_unit_none() {
    let v: std::collections::VecDeque<()> = [(); 10].iter().cloned().collect();
    let mut iter = v.into_iter();
    assert!(iter.try_fold((), |_, _| None).is_none());
    assert_eq!(iter.len(), 9);
}

#[test]
fn test_try_fold_rotated() {
    let mut v: VecDeque<_> = (0..12).collect();
    for n in 0..10 {
        if n & 1 == 0 {
            v.rotate_left(n);
        } else {
            v.rotate_right(n);
        }
        assert_eq!(Ok::<_, ()>(66), v.iter().try_fold(0, |a, b| Ok(a + b)));
    }
}

#[test]
fn test_try_fold_moves_iter() {
    let v: VecDeque<_> = [10, 20, 30, 40, 100, 60, 70, 80, 90].iter().collect();
    let mut iter = v.into_iter();
    assert_eq!(iter.try_fold(0_i8, |acc, &x| acc.checked_add(x)), None);
    assert_eq!(iter.next(), Some(&60));
}

#[test]
fn test_try_fold_exhaust_wrap() {
    let mut v = VecDeque::with_capacity(7);
    v.push_back(1);
    v.push_back(1);
    v.push_back(1);
    v.pop_front();
    v.pop_front();
    let mut iter = v.iter();
    let _ = iter.try_fold(0, |_, _| Some(1));
    assert!(iter.is_empty());
}

#[test]
fn test_try_fold_wraparound() {
    let mut v = VecDeque::with_capacity(8);
    v.push_back(7);
    v.push_back(8);
    v.push_back(9);
    v.push_front(2);
    v.push_front(1);
    let mut iter = v.iter();
    let _ = iter.find(|&&x| x == 2);
    assert_eq!(Some(&7), iter.next());
}

#[test]
fn test_try_rfold_rotated() {
    let mut v: VecDeque<_> = (0..12).collect();
    for n in 0..10 {
        if n & 1 == 0 {
            v.rotate_left(n);
        } else {
            v.rotate_right(n);
        }
        assert_eq!(Ok::<_, ()>(66), v.iter().try_rfold(0, |a, b| Ok(a + b)));
    }
}

#[test]
fn test_try_rfold_moves_iter() {
    let v: VecDeque<_> = [10, 20, 30, 40, 100, 60, 70, 80, 90].iter().collect();
    let mut iter = v.into_iter();
    assert_eq!(iter.try_rfold(0_i8, |acc, &x| acc.checked_add(x)), None);
    assert_eq!(iter.next_back(), Some(&70));
}

#[test]
fn truncate_leak() {
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

    let mut q = VecDeque::new();
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_back(D(false));
    q.push_front(D(true));
    q.push_front(D(false));
    q.push_front(D(false));

    catch_unwind(AssertUnwindSafe(|| q.truncate(1))).ok();

    assert_eq!(unsafe { DROPS }, 7);
}

#[test]
fn test_drain_leak() {
    static mut DROPS: i32 = 0;

    #[derive(Debug, PartialEq)]
    struct D(u32, bool);

    impl Drop for D {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }

            if self.1 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut v = VecDeque::new();
    v.push_back(D(4, false));
    v.push_back(D(5, false));
    v.push_back(D(6, false));
    v.push_front(D(3, false));
    v.push_front(D(2, true));
    v.push_front(D(1, false));
    v.push_front(D(0, false));

    catch_unwind(AssertUnwindSafe(|| {
        v.drain(1..=4);
    }))
    .ok();

    assert_eq!(unsafe { DROPS }, 4);
    assert_eq!(v.len(), 3);
    drop(v);
    assert_eq!(unsafe { DROPS }, 7);
}

#[test]
fn test_binary_search() {
    // Contiguous (front only) search:
    let deque: VecDeque<_> = vec![1, 2, 3, 5, 6].into();
    assert!(deque.as_slices().1.is_empty());
    assert_eq!(deque.binary_search(&3), Ok(2));
    assert_eq!(deque.binary_search(&4), Err(3));

    // Split search (both front & back non-empty):
    let mut deque: VecDeque<_> = vec![5, 6].into();
    deque.push_front(3);
    deque.push_front(2);
    deque.push_front(1);
    deque.push_back(10);
    assert!(!deque.as_slices().0.is_empty());
    assert!(!deque.as_slices().1.is_empty());
    assert_eq!(deque.binary_search(&0), Err(0));
    assert_eq!(deque.binary_search(&1), Ok(0));
    assert_eq!(deque.binary_search(&5), Ok(3));
    assert_eq!(deque.binary_search(&7), Err(5));
    assert_eq!(deque.binary_search(&20), Err(6));
}

#[test]
fn test_binary_search_by() {
    let deque: VecDeque<_> = vec![(1,), (2,), (3,), (5,), (6,)].into();

    assert_eq!(deque.binary_search_by(|&(v,)| v.cmp(&3)), Ok(2));
    assert_eq!(deque.binary_search_by(|&(v,)| v.cmp(&4)), Err(3));
}

#[test]
fn test_binary_search_by_key() {
    let deque: VecDeque<_> = vec![(1,), (2,), (3,), (5,), (6,)].into();

    assert_eq!(deque.binary_search_by_key(&3, |&(v,)| v), Ok(2));
    assert_eq!(deque.binary_search_by_key(&4, |&(v,)| v), Err(3));
}

#[test]
fn test_zero_sized_push() {
    const N: usize = 8;

    // Zero sized type
    struct Zst;

    // Test that for all possible sequences of push_front / push_back,
    // we end up with a deque of the correct size

    for len in 0..N {
        let mut tester = VecDeque::with_capacity(len);
        assert_eq!(tester.len(), 0);
        assert!(tester.capacity() >= len);
        for case in 0..(1 << len) {
            assert_eq!(tester.len(), 0);
            for bit in 0..len {
                if case & (1 << bit) != 0 {
                    tester.push_front(Zst);
                } else {
                    tester.push_back(Zst);
                }
            }
            assert_eq!(tester.len(), len);
            assert_eq!(tester.iter().count(), len);
            tester.clear();
        }
    }
}

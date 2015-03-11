// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::{SipHasher, self};

use test;

use self::Taggy::*;
use self::Taggypar::*;

#[test]
#[allow(deprecated)]
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
    debug!("{}", d[0]);
    debug!("{}", d[1]);
    debug!("{}", d[2]);
    debug!("{}", d[3]);
    assert_eq!(d[0], 1);
    assert_eq!(d[1], 2);
    assert_eq!(d[2], 3);
    assert_eq!(d[3], 4);
}

#[cfg(test)]
fn test_parameterized<T:Clone + PartialEq + Debug>(a: T, b: T, c: T, d: T) {
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

#[bench]
fn bench_new(b: &mut test::Bencher) {
    b.iter(|| {
        let ring: VecDeque<i32> = VecDeque::new();
        test::black_box(ring);
    })
}

// FIXME(japaric) privacy
/*
#[bench]
fn bench_push_back_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::with_capacity(101);
    b.iter(|| {
        for i in 0..100 {
            deq.push_back(i);
        }
        deq.head = 0;
        deq.tail = 0;
    })
}
*/

// FIXME(japaric) privacy
/*
#[bench]
fn bench_push_front_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::with_capacity(101);
    b.iter(|| {
        for i in 0..100 {
            deq.push_front(i);
        }
        deq.head = 0;
        deq.tail = 0;
    })
}
*/

// FIXME(japaric) privacy
/*
#[bench]
fn bench_pop_back_100(b: &mut test::Bencher) {
    let mut deq= VecDeque::<i32>::with_capacity(101);

    b.iter(|| {
        deq.head = 100;
        deq.tail = 0;
        while !deq.is_empty() {
            test::black_box(deq.pop_back());
        }
    })
}
*/

// FIXME(japaric) privacy
/*
#[bench]
fn bench_pop_front_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::<i32>::with_capacity(101);

    b.iter(|| {
        deq.head = 100;
        deq.tail = 0;
        while !deq.is_empty() {
            test::black_box(deq.pop_front());
        }
    })
}
*/

#[bench]
fn bench_grow_1025(b: &mut test::Bencher) {
    b.iter(|| {
        let mut deq = VecDeque::new();
        for i in 0..1025 {
            deq.push_front(i);
        }
        test::black_box(deq);
    })
}

#[bench]
fn bench_iter_1000(b: &mut test::Bencher) {
    let ring: VecDeque<_> = (0..1000).collect();

    b.iter(|| {
        let mut sum = 0;
        for &i in &ring {
            sum += i;
        }
        test::black_box(sum);
    })
}

#[bench]
fn bench_mut_iter_1000(b: &mut test::Bencher) {
    let mut ring: VecDeque<_> = (0..1000).collect();

    b.iter(|| {
        let mut sum = 0;
        for i in &mut ring {
            sum += *i;
        }
        test::black_box(sum);
    })
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
    t: Taggy
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
    test_parameterized::<Taggypar<i32>>(Onepar::<i32>(1),
                                        Twopar::<i32>(1, 2),
                                        Threepar::<i32>(1, 2, 3),
                                        Twopar::<i32>(17, 42));
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
    //good- [9, 12, 15, X, X, X, X, |6]
    //bug-  [15, 12, X, X, X, |6, X, X]
    assert_eq!(d3.pop_front(), Some(6));

    // Which leads us to the following state which
    // would be a failure case.
    //bug-  [15, 12, X, X, X, X, |X, X]
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
        let b: &[_] = &[&0,&1,&2,&3,&4];
        assert_eq!(d.iter().collect::<Vec<_>>(), b);
    }

    for i in 6..9 {
        d.push_front(i);
    }
    {
        let b: &[_] = &[&8,&7,&6,&0,&1,&2,&3,&4];
        assert_eq!(d.iter().collect::<Vec<_>>(), b);
    }

    let mut it = d.iter();
    let mut len = d.len();
    loop {
        match it.next() {
            None => break,
            _ => { len -= 1; assert_eq!(it.size_hint(), (len, Some(len))) }
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
        let b: &[_] = &[&4,&3,&2,&1,&0];
        assert_eq!(d.iter().rev().collect::<Vec<_>>(), b);
    }

    for i in 6..9 {
        d.push_front(i);
    }
    let b: &[_] = &[&4,&3,&2,&1,&0,&6,&7,&8];
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

    assert_eq!(d.iter_mut().rev().cloned().collect::<Vec<_>>(),
               vec![4, 3, 2]);
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

        let b = vec![0,1,2,3,4];
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

        let b = vec![8,7,6,0,1,2,3,4];
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
            let mut iter = d.drain();

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

        assert_eq!(d.drain().collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
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

        assert_eq!(d.drain().collect::<Vec<_>>(), [8,7,6,0,1,2,3,4]);
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
            let mut it = d.drain();
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
    use std::iter;

    let v = vec!(1,2,3,4,5,6,7);
    let deq: VecDeque<_> = v.iter().cloned().collect();
    let u: Vec<_> = deq.iter().cloned().collect();
    assert_eq!(u, v);

    let seq = iter::count(0, 2).take(256);
    let deq: VecDeque<_> = seq.collect();
    for (i, &x) in deq.iter().enumerate() {
        assert_eq!(2*i, x);
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

  assert!(hash::hash::<_, SipHasher>(&x) == hash::hash::<_, SipHasher>(&y));
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

    let ringbuf: VecDeque<_> = vec!["just", "one", "test", "more"].iter()
                                                                    .cloned()
                                                                    .collect();
    assert_eq!(format!("{:?}", ringbuf), "[\"just\", \"one\", \"test\", \"more\"]");
}

#[test]
fn test_drop() {
    static mut drops: i32 = 0;
    struct Elem;
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe { drops += 1; }
        }
    }

    let mut ring = VecDeque::new();
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.push_back(Elem);
    ring.push_front(Elem);
    drop(ring);

    assert_eq!(unsafe {drops}, 4);
}

#[test]
fn test_drop_with_pop() {
    static mut drops: i32 = 0;
    struct Elem;
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe { drops += 1; }
        }
    }

    let mut ring = VecDeque::new();
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.push_back(Elem);
    ring.push_front(Elem);

    drop(ring.pop_back());
    drop(ring.pop_front());
    assert_eq!(unsafe {drops}, 2);

    drop(ring);
    assert_eq!(unsafe {drops}, 4);
}

#[test]
fn test_drop_clear() {
    static mut drops: i32 = 0;
    struct Elem;
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe { drops += 1; }
        }
    }

    let mut ring = VecDeque::new();
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.push_back(Elem);
    ring.push_front(Elem);
    ring.clear();
    assert_eq!(unsafe {drops}, 4);

    drop(ring);
    assert_eq!(unsafe {drops}, 4);
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
        None => ()
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

// FIXME(japaric) privacy
/*
#[test]
fn test_swap_front_back_remove() {
    fn test(back: bool) {
        // This test checks that every single combination of tail position and length is tested.
        // Capacity 15 should be large enough to cover every case.
        let mut tester = VecDeque::with_capacity(15);
        let usable_cap = tester.capacity();
        let final_len = usable_cap / 2;

        for len in 0..final_len {
            let expected = if back {
                (0..len).collect()
            } else {
                (0..len).rev().collect()
            };
            for tail_pos in 0..usable_cap {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                if back {
                    for i in 0..len * 2 {
                        tester.push_front(i);
                    }
                    for i in 0..len {
                        assert_eq!(tester.swap_back_remove(i), Some(len * 2 - 1 - i));
                    }
                } else {
                    for i in 0..len * 2 {
                        tester.push_back(i);
                    }
                    for i in 0..len {
                        let idx = tester.len() - 1 - i;
                        assert_eq!(tester.swap_front_remove(idx), Some(len * 2 - 1 - i));
                    }
                }
                assert!(tester.tail < tester.cap);
                assert!(tester.head < tester.cap);
                assert_eq!(tester, expected);
            }
        }
    }
    test(true);
    test(false);
}
*/

// FIXME(japaric) privacy
/*
#[test]
fn test_insert() {
    // This test checks that every single combination of tail position, length, and
    // insertion position is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();


    // len is the length *after* insertion
    for len in 1..cap {
        // 0, 1, 2, .., len - 1
        let expected = iter::count(0, 1).take(len).collect();
        for tail_pos in 0..cap {
            for to_insert in 0..len {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                for i in 0..len {
                    if i != to_insert {
                        tester.push_back(i);
                    }
                }
                tester.insert(to_insert, to_insert);
                assert!(tester.tail < tester.cap);
                assert!(tester.head < tester.cap);
                assert_eq!(tester, expected);
            }
        }
    }
}
*/

// FIXME(japaric) privacy
/*
#[test]
fn test_remove() {
    // This test checks that every single combination of tail position, length, and
    // removal position is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();

    // len is the length *after* removal
    for len in 0..cap - 1 {
        // 0, 1, 2, .., len - 1
        let expected = iter::count(0, 1).take(len).collect();
        for tail_pos in 0..cap {
            for to_remove in 0..len + 1 {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                for i in 0..len {
                    if i == to_remove {
                        tester.push_back(1234);
                    }
                    tester.push_back(i);
                }
                if to_remove == len {
                    tester.push_back(1234);
                }
                tester.remove(to_remove);
                assert!(tester.tail < tester.cap);
                assert!(tester.head < tester.cap);
                assert_eq!(tester, expected);
            }
        }
    }
}
*/

// FIXME(japaric) privacy
/*
#[test]
fn test_shrink_to_fit() {
    // This test checks that every single combination of head and tail position,
    // is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();
    tester.reserve(63);
    let max_cap = tester.capacity();

    for len in 0..cap + 1 {
        // 0, 1, 2, .., len - 1
        let expected = iter::count(0, 1).take(len).collect();
        for tail_pos in 0..max_cap + 1 {
            tester.tail = tail_pos;
            tester.head = tail_pos;
            tester.reserve(63);
            for i in 0..len {
                tester.push_back(i);
            }
            tester.shrink_to_fit();
            assert!(tester.capacity() <= cap);
            assert!(tester.tail < tester.cap);
            assert!(tester.head < tester.cap);
            assert_eq!(tester, expected);
        }
    }
}
*/

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
    let first = cap/2;
    let last  = cap - first;
    for i in 0..first {
        ring.push_back(i);

        let (left, right) = ring.as_slices();
        let expected: Vec<_> = (0..i+1).collect();
        assert_eq!(left, expected);
        assert_eq!(right, []);
    }

    for j in -last..0 {
        ring.push_front(j);
        let (left, right) = ring.as_slices();
        let expected_left: Vec<_> = (-last..j+1).rev().collect();
        let expected_right: Vec<_> = (0..first).collect();
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);
    }

    assert_eq!(ring.len() as i32, cap);
    assert_eq!(ring.capacity() as i32, cap);
}

#[test]
fn test_as_mut_slices() {
    let mut ring: VecDeque<i32> = VecDeque::with_capacity(127);
    let cap = ring.capacity() as i32;
    let first = cap/2;
    let last  = cap - first;
    for i in 0..first {
        ring.push_back(i);

        let (left, right) = ring.as_mut_slices();
        let expected: Vec<_> = (0..i+1).collect();
        assert_eq!(left, expected);
        assert_eq!(right, []);
    }

    for j in -last..0 {
        ring.push_front(j);
        let (left, right) = ring.as_mut_slices();
        let expected_left: Vec<_> = (-last..j+1).rev().collect();
        let expected_right: Vec<_> = (0..first).collect();
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);
    }

    assert_eq!(ring.len() as i32, cap);
    assert_eq!(ring.capacity() as i32, cap);
}

// FIXME(japaric) privacy
/*
#[test]
fn test_split_off() {
    // This test checks that every single combination of tail position, length, and
    // split position is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();

    // len is the length *before* splitting
    for len in 0..cap {
        // index to split at
        for at in 0..len + 1 {
            // 0, 1, 2, .., at - 1 (may be empty)
            let expected_self = iter::count(0, 1).take(at).collect();
            // at, at + 1, .., len - 1 (may be empty)
            let expected_other = iter::count(at, 1).take(len - at).collect();

            for tail_pos in 0..cap {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                for i in 0..len {
                    tester.push_back(i);
                }
                let result = tester.split_off(at);
                assert!(tester.tail < tester.cap);
                assert!(tester.head < tester.cap);
                assert!(result.tail < result.cap);
                assert!(result.head < result.cap);
                assert_eq!(tester, expected_self);
                assert_eq!(result, expected_other);
            }
        }
    }
}
*/

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

// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering::{Equal, Greater, Less};
use std::mem;
use std::__rand::{Rng, thread_rng};
use std::rc::Rc;

fn square(n: usize) -> usize {
    n * n
}

fn is_odd(n: &usize) -> bool {
    *n % 2 == 1
}

#[test]
fn test_from_fn() {
    // Test on-stack from_fn.
    let mut v: Vec<_> = (0..3).map(square).collect();
    {
        let v = v;
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 4);
    }

    // Test on-heap from_fn.
    v = (0..5).map(square).collect();
    {
        let v = v;
        assert_eq!(v.len(), 5);
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 4);
        assert_eq!(v[3], 9);
        assert_eq!(v[4], 16);
    }
}

#[test]
fn test_from_elem() {
    // Test on-stack from_elem.
    let mut v = vec![10, 10];
    {
        let v = v;
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 10);
    }

    // Test on-heap from_elem.
    v = vec![20; 6];
    {
        let v = &v[..];
        assert_eq!(v[0], 20);
        assert_eq!(v[1], 20);
        assert_eq!(v[2], 20);
        assert_eq!(v[3], 20);
        assert_eq!(v[4], 20);
        assert_eq!(v[5], 20);
    }
}

#[test]
fn test_is_empty() {
    let xs: [i32; 0] = [];
    assert!(xs.is_empty());
    assert!(![0].is_empty());
}

#[test]
fn test_len_divzero() {
    type Z = [i8; 0];
    let v0: &[Z] = &[];
    let v1: &[Z] = &[[]];
    let v2: &[Z] = &[[], []];
    assert_eq!(mem::size_of::<Z>(), 0);
    assert_eq!(v0.len(), 0);
    assert_eq!(v1.len(), 1);
    assert_eq!(v2.len(), 2);
}

#[test]
fn test_get() {
    let mut a = vec![11];
    assert_eq!(a.get(1), None);
    a = vec![11, 12];
    assert_eq!(a.get(1).unwrap(), &12);
    a = vec![11, 12, 13];
    assert_eq!(a.get(1).unwrap(), &12);
}

#[test]
fn test_first() {
    let mut a = vec![];
    assert_eq!(a.first(), None);
    a = vec![11];
    assert_eq!(a.first().unwrap(), &11);
    a = vec![11, 12];
    assert_eq!(a.first().unwrap(), &11);
}

#[test]
fn test_first_mut() {
    let mut a = vec![];
    assert_eq!(a.first_mut(), None);
    a = vec![11];
    assert_eq!(*a.first_mut().unwrap(), 11);
    a = vec![11, 12];
    assert_eq!(*a.first_mut().unwrap(), 11);
}

#[test]
fn test_split_first() {
    let mut a = vec![11];
    let b: &[i32] = &[];
    assert!(b.split_first().is_none());
    assert_eq!(a.split_first(), Some((&11, b)));
    a = vec![11, 12];
    let b: &[i32] = &[12];
    assert_eq!(a.split_first(), Some((&11, b)));
}

#[test]
fn test_split_first_mut() {
    let mut a = vec![11];
    let b: &mut [i32] = &mut [];
    assert!(b.split_first_mut().is_none());
    assert!(a.split_first_mut() == Some((&mut 11, b)));
    a = vec![11, 12];
    let b: &mut [_] = &mut [12];
    assert!(a.split_first_mut() == Some((&mut 11, b)));
}

#[test]
fn test_split_last() {
    let mut a = vec![11];
    let b: &[i32] = &[];
    assert!(b.split_last().is_none());
    assert_eq!(a.split_last(), Some((&11, b)));
    a = vec![11, 12];
    let b: &[_] = &[11];
    assert_eq!(a.split_last(), Some((&12, b)));
}

#[test]
fn test_split_last_mut() {
    let mut a = vec![11];
    let b: &mut [i32] = &mut [];
    assert!(b.split_last_mut().is_none());
    assert!(a.split_last_mut() == Some((&mut 11, b)));

    a = vec![11, 12];
    let b: &mut [_] = &mut [11];
    assert!(a.split_last_mut() == Some((&mut 12, b)));
}

#[test]
fn test_last() {
    let mut a = vec![];
    assert_eq!(a.last(), None);
    a = vec![11];
    assert_eq!(a.last().unwrap(), &11);
    a = vec![11, 12];
    assert_eq!(a.last().unwrap(), &12);
}

#[test]
fn test_last_mut() {
    let mut a = vec![];
    assert_eq!(a.last_mut(), None);
    a = vec![11];
    assert_eq!(*a.last_mut().unwrap(), 11);
    a = vec![11, 12];
    assert_eq!(*a.last_mut().unwrap(), 12);
}

#[test]
fn test_slice() {
    // Test fixed length vector.
    let vec_fixed = [1, 2, 3, 4];
    let v_a = vec_fixed[1..vec_fixed.len()].to_vec();
    assert_eq!(v_a.len(), 3);

    assert_eq!(v_a[0], 2);
    assert_eq!(v_a[1], 3);
    assert_eq!(v_a[2], 4);

    // Test on stack.
    let vec_stack: &[_] = &[1, 2, 3];
    let v_b = vec_stack[1..3].to_vec();
    assert_eq!(v_b.len(), 2);

    assert_eq!(v_b[0], 2);
    assert_eq!(v_b[1], 3);

    // Test `Box<[T]>`
    let vec_unique = vec![1, 2, 3, 4, 5, 6];
    let v_d = vec_unique[1..6].to_vec();
    assert_eq!(v_d.len(), 5);

    assert_eq!(v_d[0], 2);
    assert_eq!(v_d[1], 3);
    assert_eq!(v_d[2], 4);
    assert_eq!(v_d[3], 5);
    assert_eq!(v_d[4], 6);
}

#[test]
fn test_slice_from() {
    let vec: &[_] = &[1, 2, 3, 4];
    assert_eq!(&vec[..], vec);
    let b: &[_] = &[3, 4];
    assert_eq!(&vec[2..], b);
    let b: &[_] = &[];
    assert_eq!(&vec[4..], b);
}

#[test]
fn test_slice_to() {
    let vec: &[_] = &[1, 2, 3, 4];
    assert_eq!(&vec[..4], vec);
    let b: &[_] = &[1, 2];
    assert_eq!(&vec[..2], b);
    let b: &[_] = &[];
    assert_eq!(&vec[..0], b);
}


#[test]
fn test_pop() {
    let mut v = vec![5];
    let e = v.pop();
    assert_eq!(v.len(), 0);
    assert_eq!(e, Some(5));
    let f = v.pop();
    assert_eq!(f, None);
    let g = v.pop();
    assert_eq!(g, None);
}

#[test]
fn test_swap_remove() {
    let mut v = vec![1, 2, 3, 4, 5];
    let mut e = v.swap_remove(0);
    assert_eq!(e, 1);
    assert_eq!(v, [5, 2, 3, 4]);
    e = v.swap_remove(3);
    assert_eq!(e, 4);
    assert_eq!(v, [5, 2, 3]);
}

#[test]
#[should_panic]
fn test_swap_remove_fail() {
    let mut v = vec![1];
    let _ = v.swap_remove(0);
    let _ = v.swap_remove(0);
}

#[test]
fn test_swap_remove_noncopyable() {
    // Tests that we don't accidentally run destructors twice.
    let mut v: Vec<Box<_>> = Vec::new();
    v.push(box 0);
    v.push(box 0);
    v.push(box 0);
    let mut _e = v.swap_remove(0);
    assert_eq!(v.len(), 2);
    _e = v.swap_remove(1);
    assert_eq!(v.len(), 1);
    _e = v.swap_remove(0);
    assert_eq!(v.len(), 0);
}

#[test]
fn test_push() {
    // Test on-stack push().
    let mut v = vec![];
    v.push(1);
    assert_eq!(v.len(), 1);
    assert_eq!(v[0], 1);

    // Test on-heap push().
    v.push(2);
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
}

#[test]
fn test_truncate() {
    let mut v: Vec<Box<_>> = vec![box 6, box 5, box 4];
    v.truncate(1);
    let v = v;
    assert_eq!(v.len(), 1);
    assert_eq!(*(v[0]), 6);
    // If the unsafe block didn't drop things properly, we blow up here.
}

#[test]
fn test_clear() {
    let mut v: Vec<Box<_>> = vec![box 6, box 5, box 4];
    v.clear();
    assert_eq!(v.len(), 0);
    // If the unsafe block didn't drop things properly, we blow up here.
}

#[test]
fn test_retain() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.retain(is_odd);
    assert_eq!(v, [1, 3, 5]);
}

#[test]
fn test_binary_search() {
    assert_eq!([1, 2, 3, 4, 5].binary_search(&5).ok(), Some(4));
    assert_eq!([1, 2, 3, 4, 5].binary_search(&4).ok(), Some(3));
    assert_eq!([1, 2, 3, 4, 5].binary_search(&3).ok(), Some(2));
    assert_eq!([1, 2, 3, 4, 5].binary_search(&2).ok(), Some(1));
    assert_eq!([1, 2, 3, 4, 5].binary_search(&1).ok(), Some(0));

    assert_eq!([2, 4, 6, 8, 10].binary_search(&1).ok(), None);
    assert_eq!([2, 4, 6, 8, 10].binary_search(&5).ok(), None);
    assert_eq!([2, 4, 6, 8, 10].binary_search(&4).ok(), Some(1));
    assert_eq!([2, 4, 6, 8, 10].binary_search(&10).ok(), Some(4));

    assert_eq!([2, 4, 6, 8].binary_search(&1).ok(), None);
    assert_eq!([2, 4, 6, 8].binary_search(&5).ok(), None);
    assert_eq!([2, 4, 6, 8].binary_search(&4).ok(), Some(1));
    assert_eq!([2, 4, 6, 8].binary_search(&8).ok(), Some(3));

    assert_eq!([2, 4, 6].binary_search(&1).ok(), None);
    assert_eq!([2, 4, 6].binary_search(&5).ok(), None);
    assert_eq!([2, 4, 6].binary_search(&4).ok(), Some(1));
    assert_eq!([2, 4, 6].binary_search(&6).ok(), Some(2));

    assert_eq!([2, 4].binary_search(&1).ok(), None);
    assert_eq!([2, 4].binary_search(&5).ok(), None);
    assert_eq!([2, 4].binary_search(&2).ok(), Some(0));
    assert_eq!([2, 4].binary_search(&4).ok(), Some(1));

    assert_eq!([2].binary_search(&1).ok(), None);
    assert_eq!([2].binary_search(&5).ok(), None);
    assert_eq!([2].binary_search(&2).ok(), Some(0));

    assert_eq!([].binary_search(&1).ok(), None);
    assert_eq!([].binary_search(&5).ok(), None);

    assert!([1, 1, 1, 1, 1].binary_search(&1).ok() != None);
    assert!([1, 1, 1, 1, 2].binary_search(&1).ok() != None);
    assert!([1, 1, 1, 2, 2].binary_search(&1).ok() != None);
    assert!([1, 1, 2, 2, 2].binary_search(&1).ok() != None);
    assert_eq!([1, 2, 2, 2, 2].binary_search(&1).ok(), Some(0));

    assert_eq!([1, 2, 3, 4, 5].binary_search(&6).ok(), None);
    assert_eq!([1, 2, 3, 4, 5].binary_search(&0).ok(), None);
}

#[test]
fn test_reverse() {
    let mut v = vec![10, 20];
    assert_eq!(v[0], 10);
    assert_eq!(v[1], 20);
    v.reverse();
    assert_eq!(v[0], 20);
    assert_eq!(v[1], 10);

    let mut v3 = Vec::<i32>::new();
    v3.reverse();
    assert!(v3.is_empty());
}

#[test]
fn test_sort() {
    for len in (2..25).chain(500..510) {
        for _ in 0..100 {
            let mut v: Vec<_> = thread_rng().gen_iter::<i32>().take(len).collect();
            let mut v1 = v.clone();

            v.sort();
            assert!(v.windows(2).all(|w| w[0] <= w[1]));

            v1.sort_by(|a, b| a.cmp(b));
            assert!(v1.windows(2).all(|w| w[0] <= w[1]));

            v1.sort_by(|a, b| b.cmp(a));
            assert!(v1.windows(2).all(|w| w[0] >= w[1]));
        }
    }

    // shouldn't panic
    let mut v: [i32; 0] = [];
    v.sort();

    let mut v = [0xDEADBEEFu64];
    v.sort();
    assert!(v == [0xDEADBEEF]);
}

#[test]
fn test_sort_stability() {
    for len in (2..25).chain(500..510) {
        for _ in 0..10 {
            let mut counts = [0; 10];

            // create a vector like [(6, 1), (5, 1), (6, 2), ...],
            // where the first item of each tuple is random, but
            // the second item represents which occurrence of that
            // number this element is, i.e. the second elements
            // will occur in sorted order.
            let mut v: Vec<_> = (0..len)
                .map(|_| {
                    let n = thread_rng().gen::<usize>() % 10;
                    counts[n] += 1;
                    (n, counts[n])
                })
                .collect();

            // only sort on the first element, so an unstable sort
            // may mix up the counts.
            v.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

            // this comparison includes the count (the second item
            // of the tuple), so elements with equal first items
            // will need to be ordered with increasing
            // counts... i.e. exactly asserting that this sort is
            // stable.
            assert!(v.windows(2).all(|w| w[0] <= w[1]));
        }
    }
}

#[test]
fn test_sort_zero_sized_type() {
    // Should not panic.
    [(); 10].sort();
    [(); 100].sort();
}

#[test]
fn test_concat() {
    let v: [Vec<i32>; 0] = [];
    let c = v.concat();
    assert_eq!(c, []);
    let d = [vec![1], vec![2, 3]].concat();
    assert_eq!(d, [1, 2, 3]);

    let v: &[&[_]] = &[&[1], &[2, 3]];
    assert_eq!(v.join(&0), [1, 0, 2, 3]);
    let v: &[&[_]] = &[&[1], &[2], &[3]];
    assert_eq!(v.join(&0), [1, 0, 2, 0, 3]);
}

#[test]
fn test_join() {
    let v: [Vec<i32>; 0] = [];
    assert_eq!(v.join(&0), []);
    assert_eq!([vec![1], vec![2, 3]].join(&0), [1, 0, 2, 3]);
    assert_eq!([vec![1], vec![2], vec![3]].join(&0), [1, 0, 2, 0, 3]);

    let v: [&[_]; 2] = [&[1], &[2, 3]];
    assert_eq!(v.join(&0), [1, 0, 2, 3]);
    let v: [&[_]; 3] = [&[1], &[2], &[3]];
    assert_eq!(v.join(&0), [1, 0, 2, 0, 3]);
}

#[test]
fn test_insert() {
    let mut a = vec![1, 2, 4];
    a.insert(2, 3);
    assert_eq!(a, [1, 2, 3, 4]);

    let mut a = vec![1, 2, 3];
    a.insert(0, 0);
    assert_eq!(a, [0, 1, 2, 3]);

    let mut a = vec![1, 2, 3];
    a.insert(3, 4);
    assert_eq!(a, [1, 2, 3, 4]);

    let mut a = vec![];
    a.insert(0, 1);
    assert_eq!(a, [1]);
}

#[test]
#[should_panic]
fn test_insert_oob() {
    let mut a = vec![1, 2, 3];
    a.insert(4, 5);
}

#[test]
fn test_remove() {
    let mut a = vec![1, 2, 3, 4];

    assert_eq!(a.remove(2), 3);
    assert_eq!(a, [1, 2, 4]);

    assert_eq!(a.remove(2), 4);
    assert_eq!(a, [1, 2]);

    assert_eq!(a.remove(0), 1);
    assert_eq!(a, [2]);

    assert_eq!(a.remove(0), 2);
    assert_eq!(a, []);
}

#[test]
#[should_panic]
fn test_remove_fail() {
    let mut a = vec![1];
    let _ = a.remove(0);
    let _ = a.remove(0);
}

#[test]
fn test_capacity() {
    let mut v = vec![0];
    v.reserve_exact(10);
    assert!(v.capacity() >= 11);
}

#[test]
fn test_slice_2() {
    let v = vec![1, 2, 3, 4, 5];
    let v = &v[1..3];
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 2);
    assert_eq!(v[1], 3);
}

macro_rules! assert_order {
    (Greater, $a:expr, $b:expr) => {
        assert_eq!($a.cmp($b), Greater);
        assert!($a > $b);
    };
    (Less, $a:expr, $b:expr) => {
        assert_eq!($a.cmp($b), Less);
        assert!($a < $b);
    };
    (Equal, $a:expr, $b:expr) => {
        assert_eq!($a.cmp($b), Equal);
        assert_eq!($a, $b);
    }
}

#[test]
fn test_total_ord_u8() {
    let c = &[1u8, 2, 3];
    assert_order!(Greater, &[1u8, 2, 3, 4][..], &c[..]);
    let c = &[1u8, 2, 3, 4];
    assert_order!(Less, &[1u8, 2, 3][..], &c[..]);
    let c = &[1u8, 2, 3, 6];
    assert_order!(Equal, &[1u8, 2, 3, 6][..], &c[..]);
    let c = &[1u8, 2, 3, 4, 5, 6];
    assert_order!(Less, &[1u8, 2, 3, 4, 5, 5, 5, 5][..], &c[..]);
    let c = &[1u8, 2, 3, 4];
    assert_order!(Greater, &[2u8, 2][..], &c[..]);
}


#[test]
fn test_total_ord_i32() {
    let c = &[1, 2, 3];
    assert_order!(Greater, &[1, 2, 3, 4][..], &c[..]);
    let c = &[1, 2, 3, 4];
    assert_order!(Less, &[1, 2, 3][..], &c[..]);
    let c = &[1, 2, 3, 6];
    assert_order!(Equal, &[1, 2, 3, 6][..], &c[..]);
    let c = &[1, 2, 3, 4, 5, 6];
    assert_order!(Less, &[1, 2, 3, 4, 5, 5, 5, 5][..], &c[..]);
    let c = &[1, 2, 3, 4];
    assert_order!(Greater, &[2, 2][..], &c[..]);
}

#[test]
fn test_iterator() {
    let xs = [1, 2, 5, 10, 11];
    let mut it = xs.iter();
    assert_eq!(it.size_hint(), (5, Some(5)));
    assert_eq!(it.next().unwrap(), &1);
    assert_eq!(it.size_hint(), (4, Some(4)));
    assert_eq!(it.next().unwrap(), &2);
    assert_eq!(it.size_hint(), (3, Some(3)));
    assert_eq!(it.next().unwrap(), &5);
    assert_eq!(it.size_hint(), (2, Some(2)));
    assert_eq!(it.next().unwrap(), &10);
    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next().unwrap(), &11);
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
}

#[test]
fn test_iter_size_hints() {
    let mut xs = [1, 2, 5, 10, 11];
    assert_eq!(xs.iter().size_hint(), (5, Some(5)));
    assert_eq!(xs.iter_mut().size_hint(), (5, Some(5)));
}

#[test]
fn test_iter_as_slice() {
    let xs = [1, 2, 5, 10, 11];
    let mut iter = xs.iter();
    assert_eq!(iter.as_slice(), &[1, 2, 5, 10, 11]);
    iter.next();
    assert_eq!(iter.as_slice(), &[2, 5, 10, 11]);
}

#[test]
fn test_iter_as_ref() {
    let xs = [1, 2, 5, 10, 11];
    let mut iter = xs.iter();
    assert_eq!(iter.as_ref(), &[1, 2, 5, 10, 11]);
    iter.next();
    assert_eq!(iter.as_ref(), &[2, 5, 10, 11]);
}

#[test]
fn test_iter_clone() {
    let xs = [1, 2, 5];
    let mut it = xs.iter();
    it.next();
    let mut jt = it.clone();
    assert_eq!(it.next(), jt.next());
    assert_eq!(it.next(), jt.next());
    assert_eq!(it.next(), jt.next());
}

#[test]
fn test_iter_is_empty() {
    let xs = [1, 2, 5, 10, 11];
    for i in 0..xs.len() {
        for j in i..xs.len() {
            assert_eq!(xs[i..j].iter().is_empty(), xs[i..j].is_empty());
        }
    }
}

#[test]
fn test_mut_iterator() {
    let mut xs = [1, 2, 3, 4, 5];
    for x in &mut xs {
        *x += 1;
    }
    assert!(xs == [2, 3, 4, 5, 6])
}

#[test]
fn test_rev_iterator() {

    let xs = [1, 2, 5, 10, 11];
    let ys = [11, 10, 5, 2, 1];
    let mut i = 0;
    for &x in xs.iter().rev() {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, 5);
}

#[test]
fn test_mut_rev_iterator() {
    let mut xs = [1, 2, 3, 4, 5];
    for (i, x) in xs.iter_mut().rev().enumerate() {
        *x += i;
    }
    assert!(xs == [5, 5, 5, 5, 5])
}

#[test]
fn test_move_iterator() {
    let xs = vec![1, 2, 3, 4, 5];
    assert_eq!(xs.into_iter().fold(0, |a: usize, b: usize| 10 * a + b),
               12345);
}

#[test]
fn test_move_rev_iterator() {
    let xs = vec![1, 2, 3, 4, 5];
    assert_eq!(xs.into_iter().rev().fold(0, |a: usize, b: usize| 10 * a + b),
               54321);
}

#[test]
fn test_splitator() {
    let xs = &[1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[1], &[3], &[5]];
    assert_eq!(xs.split(|x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[], &[2, 3, 4, 5]];
    assert_eq!(xs.split(|x| *x == 1).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4], &[]];
    assert_eq!(xs.split(|x| *x == 5).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split(|x| *x == 10).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[], &[], &[], &[], &[], &[]];
    assert_eq!(xs.split(|_| true).collect::<Vec<&[i32]>>(), splits);

    let xs: &[i32] = &[];
    let splits: &[&[i32]] = &[&[]];
    assert_eq!(xs.split(|x| *x == 5).collect::<Vec<&[i32]>>(), splits);
}

#[test]
fn test_splitnator() {
    let xs = &[1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.splitn(1, |x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1], &[3, 4, 5]];
    assert_eq!(xs.splitn(2, |x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[], &[], &[], &[4, 5]];
    assert_eq!(xs.splitn(4, |_| true).collect::<Vec<_>>(), splits);

    let xs: &[i32] = &[];
    let splits: &[&[i32]] = &[&[]];
    assert_eq!(xs.splitn(2, |x| *x == 5).collect::<Vec<_>>(), splits);
}

#[test]
fn test_splitnator_mut() {
    let xs = &mut [1, 2, 3, 4, 5];

    let splits: &[&mut [_]] = &[&mut [1, 2, 3, 4, 5]];
    assert_eq!(xs.splitn_mut(1, |x| *x % 2 == 0).collect::<Vec<_>>(),
               splits);
    let splits: &[&mut [_]] = &[&mut [1], &mut [3, 4, 5]];
    assert_eq!(xs.splitn_mut(2, |x| *x % 2 == 0).collect::<Vec<_>>(),
               splits);
    let splits: &[&mut [_]] = &[&mut [], &mut [], &mut [], &mut [4, 5]];
    assert_eq!(xs.splitn_mut(4, |_| true).collect::<Vec<_>>(), splits);

    let xs: &mut [i32] = &mut [];
    let splits: &[&mut [i32]] = &[&mut []];
    assert_eq!(xs.splitn_mut(2, |x| *x == 5).collect::<Vec<_>>(), splits);
}

#[test]
fn test_rsplitator() {
    let xs = &[1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[5], &[3], &[1]];
    assert_eq!(xs.split(|x| *x % 2 == 0).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[2, 3, 4, 5], &[]];
    assert_eq!(xs.split(|x| *x == 1).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[], &[1, 2, 3, 4]];
    assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split(|x| *x == 10).rev().collect::<Vec<_>>(), splits);

    let xs: &[i32] = &[];
    let splits: &[&[i32]] = &[&[]];
    assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<&[i32]>>(), splits);
}

#[test]
fn test_rsplitnator() {
    let xs = &[1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.rsplitn(1, |x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[5], &[1, 2, 3]];
    assert_eq!(xs.rsplitn(2, |x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[], &[], &[], &[1, 2]];
    assert_eq!(xs.rsplitn(4, |_| true).collect::<Vec<_>>(), splits);

    let xs: &[i32] = &[];
    let splits: &[&[i32]] = &[&[]];
    assert_eq!(xs.rsplitn(2, |x| *x == 5).collect::<Vec<&[i32]>>(), splits);
    assert!(xs.rsplitn(0, |x| *x % 2 == 0).next().is_none());
}

#[test]
fn test_windowsator() {
    let v = &[1, 2, 3, 4];

    let wins: &[&[_]] = &[&[1, 2], &[2, 3], &[3, 4]];
    assert_eq!(v.windows(2).collect::<Vec<_>>(), wins);

    let wins: &[&[_]] = &[&[1, 2, 3], &[2, 3, 4]];
    assert_eq!(v.windows(3).collect::<Vec<_>>(), wins);
    assert!(v.windows(6).next().is_none());

    let wins: &[&[_]] = &[&[3, 4], &[2, 3], &[1, 2]];
    assert_eq!(v.windows(2).rev().collect::<Vec<&[_]>>(), wins);
}

#[test]
#[should_panic]
fn test_windowsator_0() {
    let v = &[1, 2, 3, 4];
    let _it = v.windows(0);
}

#[test]
fn test_chunksator() {
    let v = &[1, 2, 3, 4, 5];

    assert_eq!(v.chunks(2).len(), 3);

    let chunks: &[&[_]] = &[&[1, 2], &[3, 4], &[5]];
    assert_eq!(v.chunks(2).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[&[1, 2, 3], &[4, 5]];
    assert_eq!(v.chunks(3).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(v.chunks(6).collect::<Vec<_>>(), chunks);

    let chunks: &[&[_]] = &[&[5], &[3, 4], &[1, 2]];
    assert_eq!(v.chunks(2).rev().collect::<Vec<_>>(), chunks);
}

#[test]
#[should_panic]
fn test_chunksator_0() {
    let v = &[1, 2, 3, 4];
    let _it = v.chunks(0);
}

#[test]
fn test_reverse_part() {
    let mut values = [1, 2, 3, 4, 5];
    values[1..4].reverse();
    assert!(values == [1, 4, 3, 2, 5]);
}

#[test]
fn test_show() {
    macro_rules! test_show_vec {
        ($x:expr, $x_str:expr) => ({
            let (x, x_str) = ($x, $x_str);
            assert_eq!(format!("{:?}", x), x_str);
            assert_eq!(format!("{:?}", x), x_str);
        })
    }
    let empty = Vec::<i32>::new();
    test_show_vec!(empty, "[]");
    test_show_vec!(vec![1], "[1]");
    test_show_vec!(vec![1, 2, 3], "[1, 2, 3]");
    test_show_vec!(vec![vec![], vec![1], vec![1, 1]], "[[], [1], [1, 1]]");

    let empty_mut: &mut [i32] = &mut [];
    test_show_vec!(empty_mut, "[]");
    let v = &mut [1];
    test_show_vec!(v, "[1]");
    let v = &mut [1, 2, 3];
    test_show_vec!(v, "[1, 2, 3]");
    let v: &mut [&mut [_]] = &mut [&mut [], &mut [1], &mut [1, 1]];
    test_show_vec!(v, "[[], [1], [1, 1]]");
}

#[test]
fn test_vec_default() {
    macro_rules! t {
        ($ty:ty) => {{
            let v: $ty = Default::default();
            assert!(v.is_empty());
        }}
    }

    t!(&[i32]);
    t!(Vec<i32>);
}

#[test]
#[should_panic]
fn test_overflow_does_not_cause_segfault() {
    let mut v = vec![];
    v.reserve_exact(!0);
    v.push(1);
    v.push(2);
}

#[test]
#[should_panic]
fn test_overflow_does_not_cause_segfault_managed() {
    let mut v = vec![Rc::new(1)];
    v.reserve_exact(!0);
    v.push(Rc::new(2));
}

#[test]
fn test_mut_split_at() {
    let mut values = [1, 2, 3, 4, 5];
    {
        let (left, right) = values.split_at_mut(2);
        {
            let left: &[_] = left;
            assert!(left[..left.len()] == [1, 2]);
        }
        for p in left {
            *p += 1;
        }

        {
            let right: &[_] = right;
            assert!(right[..right.len()] == [3, 4, 5]);
        }
        for p in right {
            *p += 2;
        }
    }

    assert!(values == [2, 3, 5, 6, 7]);
}

#[derive(Clone, PartialEq)]
struct Foo;

#[test]
fn test_iter_zero_sized() {
    let mut v = vec![Foo, Foo, Foo];
    assert_eq!(v.len(), 3);
    let mut cnt = 0;

    for f in &v {
        assert!(*f == Foo);
        cnt += 1;
    }
    assert_eq!(cnt, 3);

    for f in &v[1..3] {
        assert!(*f == Foo);
        cnt += 1;
    }
    assert_eq!(cnt, 5);

    for f in &mut v {
        assert!(*f == Foo);
        cnt += 1;
    }
    assert_eq!(cnt, 8);

    for f in v {
        assert!(f == Foo);
        cnt += 1;
    }
    assert_eq!(cnt, 11);

    let xs: [Foo; 3] = [Foo, Foo, Foo];
    cnt = 0;
    for f in &xs {
        assert!(*f == Foo);
        cnt += 1;
    }
    assert!(cnt == 3);
}

#[test]
fn test_shrink_to_fit() {
    let mut xs = vec![0, 1, 2, 3];
    for i in 4..100 {
        xs.push(i)
    }
    assert_eq!(xs.capacity(), 128);
    xs.shrink_to_fit();
    assert_eq!(xs.capacity(), 100);
    assert_eq!(xs, (0..100).collect::<Vec<_>>());
}

#[test]
fn test_starts_with() {
    assert!(b"foobar".starts_with(b"foo"));
    assert!(!b"foobar".starts_with(b"oob"));
    assert!(!b"foobar".starts_with(b"bar"));
    assert!(!b"foo".starts_with(b"foobar"));
    assert!(!b"bar".starts_with(b"foobar"));
    assert!(b"foobar".starts_with(b"foobar"));
    let empty: &[u8] = &[];
    assert!(empty.starts_with(empty));
    assert!(!empty.starts_with(b"foo"));
    assert!(b"foobar".starts_with(empty));
}

#[test]
fn test_ends_with() {
    assert!(b"foobar".ends_with(b"bar"));
    assert!(!b"foobar".ends_with(b"oba"));
    assert!(!b"foobar".ends_with(b"foo"));
    assert!(!b"foo".ends_with(b"foobar"));
    assert!(!b"bar".ends_with(b"foobar"));
    assert!(b"foobar".ends_with(b"foobar"));
    let empty: &[u8] = &[];
    assert!(empty.ends_with(empty));
    assert!(!empty.ends_with(b"foo"));
    assert!(b"foobar".ends_with(empty));
}

#[test]
fn test_mut_splitator() {
    let mut xs = [0, 1, 0, 2, 3, 0, 0, 4, 5, 0];
    assert_eq!(xs.split_mut(|x| *x == 0).count(), 6);
    for slice in xs.split_mut(|x| *x == 0) {
        slice.reverse();
    }
    assert!(xs == [0, 1, 0, 3, 2, 0, 0, 5, 4, 0]);

    let mut xs = [0, 1, 0, 2, 3, 0, 0, 4, 5, 0, 6, 7];
    for slice in xs.split_mut(|x| *x == 0).take(5) {
        slice.reverse();
    }
    assert!(xs == [0, 1, 0, 3, 2, 0, 0, 5, 4, 0, 6, 7]);
}

#[test]
fn test_mut_splitator_rev() {
    let mut xs = [1, 2, 0, 3, 4, 0, 0, 5, 6, 0];
    for slice in xs.split_mut(|x| *x == 0).rev().take(4) {
        slice.reverse();
    }
    assert!(xs == [1, 2, 0, 4, 3, 0, 0, 6, 5, 0]);
}

#[test]
fn test_get_mut() {
    let mut v = [0, 1, 2];
    assert_eq!(v.get_mut(3), None);
    v.get_mut(1).map(|e| *e = 7);
    assert_eq!(v[1], 7);
    let mut x = 2;
    assert_eq!(v.get_mut(2), Some(&mut x));
}

#[test]
fn test_mut_chunks() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    assert_eq!(v.chunks_mut(2).len(), 4);
    for (i, chunk) in v.chunks_mut(3).enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [0, 0, 0, 1, 1, 1, 2];
    assert!(v == result);
}

#[test]
fn test_mut_chunks_rev() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    for (i, chunk) in v.chunks_mut(3).rev().enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [2, 2, 2, 1, 1, 1, 0];
    assert!(v == result);
}

#[test]
#[should_panic]
fn test_mut_chunks_0() {
    let mut v = [1, 2, 3, 4];
    let _it = v.chunks_mut(0);
}

#[test]
fn test_mut_last() {
    let mut x = [1, 2, 3, 4, 5];
    let h = x.last_mut();
    assert_eq!(*h.unwrap(), 5);

    let y: &mut [i32] = &mut [];
    assert!(y.last_mut().is_none());
}

#[test]
fn test_to_vec() {
    let xs: Box<_> = box [1, 2, 3];
    let ys = xs.to_vec();
    assert_eq!(ys, [1, 2, 3]);
}

#[test]
fn test_box_slice_clone() {
    let data = vec![vec![0, 1], vec![0], vec![1]];
    let data2 = data.clone().into_boxed_slice().clone().to_vec();

    assert_eq!(data, data2);
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_box_slice_clone_panics() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::spawn;

    struct Canary {
        count: Arc<AtomicUsize>,
        panics: bool,
    }

    impl Drop for Canary {
        fn drop(&mut self) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Clone for Canary {
        fn clone(&self) -> Self {
            if self.panics {
                panic!()
            }

            Canary {
                count: self.count.clone(),
                panics: self.panics,
            }
        }
    }

    let drop_count = Arc::new(AtomicUsize::new(0));
    let canary = Canary {
        count: drop_count.clone(),
        panics: false,
    };
    let panic = Canary {
        count: drop_count.clone(),
        panics: true,
    };

    spawn(move || {
            // When xs is dropped, +5.
            let xs = vec![canary.clone(), canary.clone(), canary.clone(), panic, canary]
                .into_boxed_slice();

            // When panic is cloned, +3.
            xs.clone();
        })
        .join()
        .unwrap_err();

    // Total = 8
    assert_eq!(drop_count.load(Ordering::SeqCst), 8);
}

#[test]
fn test_copy_from_slice() {
    let src = [0, 1, 2, 3, 4, 5];
    let mut dst = [0; 6];
    dst.copy_from_slice(&src);
    assert_eq!(src, dst)
}

#[test]
#[should_panic(expected = "destination and source slices have different lengths")]
fn test_copy_from_slice_dst_longer() {
    let src = [0, 1, 2, 3];
    let mut dst = [0; 5];
    dst.copy_from_slice(&src);
}

#[test]
#[should_panic(expected = "destination and source slices have different lengths")]
fn test_copy_from_slice_dst_shorter() {
    let src = [0, 1, 2, 3];
    let mut dst = [0; 3];
    dst.copy_from_slice(&src);
}

mod bench {
    use std::{mem, ptr};
    use std::__rand::{Rng, thread_rng};

    use test::{Bencher, black_box};

    #[bench]
    fn iterator(b: &mut Bencher) {
        // peculiar numbers to stop LLVM from optimising the summation
        // out.
        let v: Vec<_> = (0..100).map(|i| i ^ (i << 1) ^ (i >> 1)).collect();

        b.iter(|| {
            let mut sum = 0;
            for x in &v {
                sum += *x;
            }
            // sum == 11806, to stop dead code elimination.
            if sum == 0 {
                panic!()
            }
        })
    }

    #[bench]
    fn mut_iterator(b: &mut Bencher) {
        let mut v = vec![0; 100];

        b.iter(|| {
            let mut i = 0;
            for x in &mut v {
                *x = i;
                i += 1;
            }
        })
    }

    #[bench]
    fn concat(b: &mut Bencher) {
        let xss: Vec<Vec<i32>> = (0..100).map(|i| (0..i).collect()).collect();
        b.iter(|| {
            xss.concat();
        });
    }

    #[bench]
    fn join(b: &mut Bencher) {
        let xss: Vec<Vec<i32>> = (0..100).map(|i| (0..i).collect()).collect();
        b.iter(|| xss.join(&0));
    }

    #[bench]
    fn push(b: &mut Bencher) {
        let mut vec = Vec::<i32>::new();
        b.iter(|| {
            vec.push(0);
            black_box(&vec);
        });
    }

    #[bench]
    fn starts_with_same_vector(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        b.iter(|| vec.starts_with(&vec))
    }

    #[bench]
    fn starts_with_single_element(b: &mut Bencher) {
        let vec: Vec<_> = vec![0];
        b.iter(|| vec.starts_with(&vec))
    }

    #[bench]
    fn starts_with_diff_one_element_at_end(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        let mut match_vec: Vec<_> = (0..99).collect();
        match_vec.push(0);
        b.iter(|| vec.starts_with(&match_vec))
    }

    #[bench]
    fn ends_with_same_vector(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        b.iter(|| vec.ends_with(&vec))
    }

    #[bench]
    fn ends_with_single_element(b: &mut Bencher) {
        let vec: Vec<_> = vec![0];
        b.iter(|| vec.ends_with(&vec))
    }

    #[bench]
    fn ends_with_diff_one_element_at_beginning(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        let mut match_vec: Vec<_> = (0..100).collect();
        match_vec[0] = 200;
        b.iter(|| vec.starts_with(&match_vec))
    }

    #[bench]
    fn contains_last_element(b: &mut Bencher) {
        let vec: Vec<_> = (0..100).collect();
        b.iter(|| vec.contains(&99))
    }

    #[bench]
    fn zero_1kb_from_elem(b: &mut Bencher) {
        b.iter(|| vec![0u8; 1024]);
    }

    #[bench]
    fn zero_1kb_set_memory(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::<u8>::with_capacity(1024);
            unsafe {
                let vp = v.as_mut_ptr();
                ptr::write_bytes(vp, 0, 1024);
                v.set_len(1024);
            }
            v
        });
    }

    #[bench]
    fn zero_1kb_loop_set(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::<u8>::with_capacity(1024);
            unsafe {
                v.set_len(1024);
            }
            for i in 0..1024 {
                v[i] = 0;
            }
        });
    }

    #[bench]
    fn zero_1kb_mut_iter(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::<u8>::with_capacity(1024);
            unsafe {
                v.set_len(1024);
            }
            for x in &mut v {
                *x = 0;
            }
            v
        });
    }

    #[bench]
    fn random_inserts(b: &mut Bencher) {
        let mut rng = thread_rng();
        b.iter(|| {
            let mut v = vec![(0, 0); 30];
            for _ in 0..100 {
                let l = v.len();
                v.insert(rng.gen::<usize>() % (l + 1), (1, 1));
            }
        })
    }
    #[bench]
    fn random_removes(b: &mut Bencher) {
        let mut rng = thread_rng();
        b.iter(|| {
            let mut v = vec![(0, 0); 130];
            for _ in 0..100 {
                let l = v.len();
                v.remove(rng.gen::<usize>() % l);
            }
        })
    }

    fn gen_ascending(len: usize) -> Vec<u64> {
        (0..len as u64).collect()
    }

    fn gen_descending(len: usize) -> Vec<u64> {
        (0..len as u64).rev().collect()
    }

    fn gen_random(len: usize) -> Vec<u64> {
        let mut rng = thread_rng();
        rng.gen_iter::<u64>().take(len).collect()
    }

    fn gen_mostly_ascending(len: usize) -> Vec<u64> {
        let mut rng = thread_rng();
        let mut v = gen_ascending(len);
        for _ in (0usize..).take_while(|x| x * x <= len) {
            let x = rng.gen::<usize>() % len;
            let y = rng.gen::<usize>() % len;
            v.swap(x, y);
        }
        v
    }

    fn gen_mostly_descending(len: usize) -> Vec<u64> {
        let mut rng = thread_rng();
        let mut v = gen_descending(len);
        for _ in (0usize..).take_while(|x| x * x <= len) {
            let x = rng.gen::<usize>() % len;
            let y = rng.gen::<usize>() % len;
            v.swap(x, y);
        }
        v
    }

    fn gen_big_random(len: usize) -> Vec<[u64; 16]> {
        let mut rng = thread_rng();
        rng.gen_iter().map(|x| [x; 16]).take(len).collect()
    }

    fn gen_big_ascending(len: usize) -> Vec<[u64; 16]> {
        (0..len as u64).map(|x| [x; 16]).take(len).collect()
    }

    fn gen_big_descending(len: usize) -> Vec<[u64; 16]> {
        (0..len as u64).rev().map(|x| [x; 16]).take(len).collect()
    }

    macro_rules! sort_bench {
        ($name:ident, $gen:expr, $len:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                b.iter(|| $gen($len).sort());
                b.bytes = $len * mem::size_of_val(&$gen(1)[0]) as u64;
            }
        }
    }

    sort_bench!(sort_small_random, gen_random, 10);
    sort_bench!(sort_small_ascending, gen_ascending, 10);
    sort_bench!(sort_small_descending, gen_descending, 10);

    sort_bench!(sort_small_big_random, gen_big_random, 10);
    sort_bench!(sort_small_big_ascending, gen_big_ascending, 10);
    sort_bench!(sort_small_big_descending, gen_big_descending, 10);

    sort_bench!(sort_medium_random, gen_random, 100);
    sort_bench!(sort_medium_ascending, gen_ascending, 100);
    sort_bench!(sort_medium_descending, gen_descending, 100);

    sort_bench!(sort_large_random, gen_random, 10000);
    sort_bench!(sort_large_ascending, gen_ascending, 10000);
    sort_bench!(sort_large_descending, gen_descending, 10000);
    sort_bench!(sort_large_mostly_ascending, gen_mostly_ascending, 10000);
    sort_bench!(sort_large_mostly_descending, gen_mostly_descending, 10000);

    sort_bench!(sort_large_big_random, gen_big_random, 10000);
    sort_bench!(sort_large_big_ascending, gen_big_ascending, 10000);
    sort_bench!(sort_large_big_descending, gen_big_descending, 10000);

    #[bench]
    fn sort_large_random_expensive(b: &mut Bencher) {
        let len = 10000;
        b.iter(|| {
            let mut count = 0;
            let cmp = move |a: &u64, b: &u64| {
                count += 1;
                if count % 1_000_000_000 == 0 {
                    panic!("should not happen");
                }
                (*a as f64).cos().partial_cmp(&(*b as f64).cos()).unwrap()
            };

            let mut v = gen_random(len);
            v.sort_by(cmp);

            black_box(count);
        });
        b.bytes = len as u64 * mem::size_of::<u64>() as u64;
    }
}

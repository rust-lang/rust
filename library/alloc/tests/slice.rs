use std::cell::Cell;
use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::convert::identity;
use std::fmt;
use std::mem;
use std::panic;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use rand::distributions::Standard;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, RngCore};

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

    // check the 1-byte-types path
    let mut v = (-50..51i8).collect::<Vec<_>>();
    v.reverse();
    assert_eq!(v, (-50..51i8).rev().collect::<Vec<_>>());

    // check the 2-byte-types path
    let mut v = (-50..51i16).collect::<Vec<_>>();
    v.reverse();
    assert_eq!(v, (-50..51i16).rev().collect::<Vec<_>>());
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_sort() {
    let mut rng = thread_rng();

    for len in (2..25).chain(500..510) {
        for &modulus in &[5, 10, 100, 1000] {
            for _ in 0..10 {
                let orig: Vec<_> =
                    rng.sample_iter::<i32, _>(&Standard).map(|x| x % modulus).take(len).collect();

                // Sort in default order.
                let mut v = orig.clone();
                v.sort();
                assert!(v.windows(2).all(|w| w[0] <= w[1]));

                // Sort in ascending order.
                let mut v = orig.clone();
                v.sort_by(|a, b| a.cmp(b));
                assert!(v.windows(2).all(|w| w[0] <= w[1]));

                // Sort in descending order.
                let mut v = orig.clone();
                v.sort_by(|a, b| b.cmp(a));
                assert!(v.windows(2).all(|w| w[0] >= w[1]));

                // Sort in lexicographic order.
                let mut v1 = orig.clone();
                let mut v2 = orig.clone();
                v1.sort_by_key(|x| x.to_string());
                v2.sort_by_cached_key(|x| x.to_string());
                assert!(v1.windows(2).all(|w| w[0].to_string() <= w[1].to_string()));
                assert!(v1 == v2);

                // Sort with many pre-sorted runs.
                let mut v = orig.clone();
                v.sort();
                v.reverse();
                for _ in 0..5 {
                    let a = rng.gen::<usize>() % len;
                    let b = rng.gen::<usize>() % len;
                    if a < b {
                        v[a..b].reverse();
                    } else {
                        v.swap(a, b);
                    }
                }
                v.sort();
                assert!(v.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }

    // Sort using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    let mut v = [0; 500];
    for i in 0..v.len() {
        v[i] = i as i32;
    }
    v.sort_by(|_, _| *[Less, Equal, Greater].choose(&mut rng).unwrap());
    v.sort();
    for i in 0..v.len() {
        assert_eq!(v[i], i as i32);
    }

    // Should not panic.
    [0i32; 0].sort();
    [(); 10].sort();
    [(); 100].sort();

    let mut v = [0xDEADBEEFu64];
    v.sort();
    assert!(v == [0xDEADBEEF]);
}

#[test]
fn test_sort_stability() {
    // Miri is too slow
    let large_range = if cfg!(miri) { 0..0 } else { 500..510 };
    let rounds = if cfg!(miri) { 1 } else { 10 };

    for len in (2..25).chain(large_range) {
        for _ in 0..rounds {
            let mut counts = [0; 10];

            // create a vector like [(6, 1), (5, 1), (6, 2), ...],
            // where the first item of each tuple is random, but
            // the second item represents which occurrence of that
            // number this element is, i.e., the second elements
            // will occur in sorted order.
            let orig: Vec<_> = (0..len)
                .map(|_| {
                    let n = thread_rng().gen::<usize>() % 10;
                    counts[n] += 1;
                    (n, counts[n])
                })
                .collect();

            let mut v = orig.clone();
            // Only sort on the first element, so an unstable sort
            // may mix up the counts.
            v.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

            // This comparison includes the count (the second item
            // of the tuple), so elements with equal first items
            // will need to be ordered with increasing
            // counts... i.e., exactly asserting that this sort is
            // stable.
            assert!(v.windows(2).all(|w| w[0] <= w[1]));

            let mut v = orig.clone();
            v.sort_by_cached_key(|&(x, _)| x);
            assert!(v.windows(2).all(|w| w[0] <= w[1]));
        }
    }
}

#[test]
fn test_rotate_left() {
    let expected: Vec<_> = (0..13).collect();
    let mut v = Vec::new();

    // no-ops
    v.clone_from(&expected);
    v.rotate_left(0);
    assert_eq!(v, expected);
    v.rotate_left(expected.len());
    assert_eq!(v, expected);
    let mut zst_array = [(), (), ()];
    zst_array.rotate_left(2);

    // happy path
    v = (5..13).chain(0..5).collect();
    v.rotate_left(8);
    assert_eq!(v, expected);

    let expected: Vec<_> = (0..1000).collect();

    // small rotations in large slice, uses ptr::copy
    v = (2..1000).chain(0..2).collect();
    v.rotate_left(998);
    assert_eq!(v, expected);
    v = (998..1000).chain(0..998).collect();
    v.rotate_left(2);
    assert_eq!(v, expected);

    // non-small prime rotation, has a few rounds of swapping
    v = (389..1000).chain(0..389).collect();
    v.rotate_left(1000 - 389);
    assert_eq!(v, expected);
}

#[test]
fn test_rotate_right() {
    let expected: Vec<_> = (0..13).collect();
    let mut v = Vec::new();

    // no-ops
    v.clone_from(&expected);
    v.rotate_right(0);
    assert_eq!(v, expected);
    v.rotate_right(expected.len());
    assert_eq!(v, expected);
    let mut zst_array = [(), (), ()];
    zst_array.rotate_right(2);

    // happy path
    v = (5..13).chain(0..5).collect();
    v.rotate_right(5);
    assert_eq!(v, expected);

    let expected: Vec<_> = (0..1000).collect();

    // small rotations in large slice, uses ptr::copy
    v = (2..1000).chain(0..2).collect();
    v.rotate_right(2);
    assert_eq!(v, expected);
    v = (998..1000).chain(0..998).collect();
    v.rotate_right(998);
    assert_eq!(v, expected);

    // non-small prime rotation, has a few rounds of swapping
    v = (389..1000).chain(0..389).collect();
    v.rotate_right(389);
    assert_eq!(v, expected);
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
fn test_join_nocopy() {
    let v: [String; 0] = [];
    assert_eq!(v.join(","), "");
    assert_eq!(["a".to_string(), "ab".into()].join(","), "a,ab");
    assert_eq!(["a".to_string(), "ab".into(), "abc".into()].join(","), "a,ab,abc");
    assert_eq!(["a".to_string(), "ab".into(), "".into()].join(","), "a,ab,");
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
    };
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
    assert_eq!(xs.into_iter().fold(0, |a: usize, b: usize| 10 * a + b), 12345);
}

#[test]
fn test_move_rev_iterator() {
    let xs = vec![1, 2, 3, 4, 5];
    assert_eq!(xs.into_iter().rev().fold(0, |a: usize, b: usize| 10 * a + b), 54321);
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
fn test_splitator_inclusive() {
    let xs = &[1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[1, 2], &[3, 4], &[5]];
    assert_eq!(xs.split_inclusive(|x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1], &[2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive(|x| *x == 1).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive(|x| *x == 5).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive(|x| *x == 10).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1], &[2], &[3], &[4], &[5]];
    assert_eq!(xs.split_inclusive(|_| true).collect::<Vec<&[i32]>>(), splits);

    let xs: &[i32] = &[];
    let splits: &[&[i32]] = &[];
    assert_eq!(xs.split_inclusive(|x| *x == 5).collect::<Vec<&[i32]>>(), splits);
}

#[test]
fn test_splitator_inclusive_reverse() {
    let xs = &[1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[5], &[3, 4], &[1, 2]];
    assert_eq!(xs.split_inclusive(|x| *x % 2 == 0).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[2, 3, 4, 5], &[1]];
    assert_eq!(xs.split_inclusive(|x| *x == 1).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive(|x| *x == 5).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive(|x| *x == 10).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[5], &[4], &[3], &[2], &[1]];
    assert_eq!(xs.split_inclusive(|_| true).rev().collect::<Vec<_>>(), splits);

    let xs: &[i32] = &[];
    let splits: &[&[i32]] = &[];
    assert_eq!(xs.split_inclusive(|x| *x == 5).rev().collect::<Vec<_>>(), splits);
}

#[test]
fn test_splitator_mut_inclusive() {
    let xs = &mut [1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[1, 2], &[3, 4], &[5]];
    assert_eq!(xs.split_inclusive_mut(|x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1], &[2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 1).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 5).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 10).collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1], &[2], &[3], &[4], &[5]];
    assert_eq!(xs.split_inclusive_mut(|_| true).collect::<Vec<_>>(), splits);

    let xs: &mut [i32] = &mut [];
    let splits: &[&[i32]] = &[];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 5).collect::<Vec<_>>(), splits);
}

#[test]
fn test_splitator_mut_inclusive_reverse() {
    let xs = &mut [1, 2, 3, 4, 5];

    let splits: &[&[_]] = &[&[5], &[3, 4], &[1, 2]];
    assert_eq!(xs.split_inclusive_mut(|x| *x % 2 == 0).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[2, 3, 4, 5], &[1]];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 1).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 5).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 10).rev().collect::<Vec<_>>(), splits);
    let splits: &[&[_]] = &[&[5], &[4], &[3], &[2], &[1]];
    assert_eq!(xs.split_inclusive_mut(|_| true).rev().collect::<Vec<_>>(), splits);

    let xs: &mut [i32] = &mut [];
    let splits: &[&[i32]] = &[];
    assert_eq!(xs.split_inclusive_mut(|x| *x == 5).rev().collect::<Vec<_>>(), splits);
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
    assert_eq!(xs.splitn_mut(1, |x| *x % 2 == 0).collect::<Vec<_>>(), splits);
    let splits: &[&mut [_]] = &[&mut [1], &mut [3, 4, 5]];
    assert_eq!(xs.splitn_mut(2, |x| *x % 2 == 0).collect::<Vec<_>>(), splits);
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
fn test_split_iterators_size_hint() {
    #[derive(Copy, Clone)]
    enum Bounds {
        Lower,
        Upper,
    }
    fn assert_tight_size_hints(mut it: impl Iterator, which: Bounds, ctx: impl fmt::Display) {
        match which {
            Bounds::Lower => {
                let mut lower_bounds = vec![it.size_hint().0];
                while let Some(_) = it.next() {
                    lower_bounds.push(it.size_hint().0);
                }
                let target: Vec<_> = (0..lower_bounds.len()).rev().collect();
                assert_eq!(lower_bounds, target, "lower bounds incorrect or not tight: {}", ctx);
            }
            Bounds::Upper => {
                let mut upper_bounds = vec![it.size_hint().1];
                while let Some(_) = it.next() {
                    upper_bounds.push(it.size_hint().1);
                }
                let target: Vec<_> = (0..upper_bounds.len()).map(Some).rev().collect();
                assert_eq!(upper_bounds, target, "upper bounds incorrect or not tight: {}", ctx);
            }
        }
    }

    for len in 0..=2 {
        let mut v: Vec<u8> = (0..len).collect();

        // p: predicate, b: bound selection
        for (p, b) in [
            // with a predicate always returning false, the split*-iterators
            // become maximally short, so the size_hint lower bounds are tight
            ((|_| false) as fn(&_) -> _, Bounds::Lower),
            // with a predicate always returning true, the split*-iterators
            // become maximally long, so the size_hint upper bounds are tight
            ((|_| true) as fn(&_) -> _, Bounds::Upper),
        ] {
            use assert_tight_size_hints as a;
            use format_args as f;

            a(v.split(p), b, "split");
            a(v.split_mut(p), b, "split_mut");
            a(v.split_inclusive(p), b, "split_inclusive");
            a(v.split_inclusive_mut(p), b, "split_inclusive_mut");
            a(v.rsplit(p), b, "rsplit");
            a(v.rsplit_mut(p), b, "rsplit_mut");

            for n in 0..=3 {
                a(v.splitn(n, p), b, f!("splitn, n = {}", n));
                a(v.splitn_mut(n, p), b, f!("splitn_mut, n = {}", n));
                a(v.rsplitn(n, p), b, f!("rsplitn, n = {}", n));
                a(v.rsplitn_mut(n, p), b, f!("rsplitn_mut, n = {}", n));
            }
        }
    }
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
fn test_chunks_exactator() {
    let v = &[1, 2, 3, 4, 5];

    assert_eq!(v.chunks_exact(2).len(), 2);

    let chunks: &[&[_]] = &[&[1, 2], &[3, 4]];
    assert_eq!(v.chunks_exact(2).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[&[1, 2, 3]];
    assert_eq!(v.chunks_exact(3).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[];
    assert_eq!(v.chunks_exact(6).collect::<Vec<_>>(), chunks);

    let chunks: &[&[_]] = &[&[3, 4], &[1, 2]];
    assert_eq!(v.chunks_exact(2).rev().collect::<Vec<_>>(), chunks);
}

#[test]
#[should_panic]
fn test_chunks_exactator_0() {
    let v = &[1, 2, 3, 4];
    let _it = v.chunks_exact(0);
}

#[test]
fn test_rchunksator() {
    let v = &[1, 2, 3, 4, 5];

    assert_eq!(v.rchunks(2).len(), 3);

    let chunks: &[&[_]] = &[&[4, 5], &[2, 3], &[1]];
    assert_eq!(v.rchunks(2).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[&[3, 4, 5], &[1, 2]];
    assert_eq!(v.rchunks(3).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[&[1, 2, 3, 4, 5]];
    assert_eq!(v.rchunks(6).collect::<Vec<_>>(), chunks);

    let chunks: &[&[_]] = &[&[1], &[2, 3], &[4, 5]];
    assert_eq!(v.rchunks(2).rev().collect::<Vec<_>>(), chunks);
}

#[test]
#[should_panic]
fn test_rchunksator_0() {
    let v = &[1, 2, 3, 4];
    let _it = v.rchunks(0);
}

#[test]
fn test_rchunks_exactator() {
    let v = &[1, 2, 3, 4, 5];

    assert_eq!(v.rchunks_exact(2).len(), 2);

    let chunks: &[&[_]] = &[&[4, 5], &[2, 3]];
    assert_eq!(v.rchunks_exact(2).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[&[3, 4, 5]];
    assert_eq!(v.rchunks_exact(3).collect::<Vec<_>>(), chunks);
    let chunks: &[&[_]] = &[];
    assert_eq!(v.rchunks_exact(6).collect::<Vec<_>>(), chunks);

    let chunks: &[&[_]] = &[&[2, 3], &[4, 5]];
    assert_eq!(v.rchunks_exact(2).rev().collect::<Vec<_>>(), chunks);
}

#[test]
#[should_panic]
fn test_rchunks_exactator_0() {
    let v = &[1, 2, 3, 4];
    let _it = v.rchunks_exact(0);
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
        ($x:expr, $x_str:expr) => {{
            let (x, x_str) = ($x, $x_str);
            assert_eq!(format!("{:?}", x), x_str);
            assert_eq!(format!("{:?}", x), x_str);
        }};
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
        }};
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
    assert_eq!(v.chunks_mut(3).len(), 3);
    for (i, chunk) in v.chunks_mut(3).enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [0, 0, 0, 1, 1, 1, 2];
    assert_eq!(v, result);
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
    assert_eq!(v, result);
}

#[test]
#[should_panic]
fn test_mut_chunks_0() {
    let mut v = [1, 2, 3, 4];
    let _it = v.chunks_mut(0);
}

#[test]
fn test_mut_chunks_exact() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    assert_eq!(v.chunks_exact_mut(3).len(), 2);
    for (i, chunk) in v.chunks_exact_mut(3).enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [0, 0, 0, 1, 1, 1, 6];
    assert_eq!(v, result);
}

#[test]
fn test_mut_chunks_exact_rev() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    for (i, chunk) in v.chunks_exact_mut(3).rev().enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [1, 1, 1, 0, 0, 0, 6];
    assert_eq!(v, result);
}

#[test]
#[should_panic]
fn test_mut_chunks_exact_0() {
    let mut v = [1, 2, 3, 4];
    let _it = v.chunks_exact_mut(0);
}

#[test]
fn test_mut_rchunks() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    assert_eq!(v.rchunks_mut(3).len(), 3);
    for (i, chunk) in v.rchunks_mut(3).enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [2, 1, 1, 1, 0, 0, 0];
    assert_eq!(v, result);
}

#[test]
fn test_mut_rchunks_rev() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    for (i, chunk) in v.rchunks_mut(3).rev().enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [0, 1, 1, 1, 2, 2, 2];
    assert_eq!(v, result);
}

#[test]
#[should_panic]
fn test_mut_rchunks_0() {
    let mut v = [1, 2, 3, 4];
    let _it = v.rchunks_mut(0);
}

#[test]
fn test_mut_rchunks_exact() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    assert_eq!(v.rchunks_exact_mut(3).len(), 2);
    for (i, chunk) in v.rchunks_exact_mut(3).enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [0, 1, 1, 1, 0, 0, 0];
    assert_eq!(v, result);
}

#[test]
fn test_mut_rchunks_exact_rev() {
    let mut v = [0, 1, 2, 3, 4, 5, 6];
    for (i, chunk) in v.rchunks_exact_mut(3).rev().enumerate() {
        for x in chunk {
            *x = i as u8;
        }
    }
    let result = [0, 0, 0, 0, 1, 1, 1];
    assert_eq!(v, result);
}

#[test]
#[should_panic]
fn test_mut_rchunks_exact_0() {
    let mut v = [1, 2, 3, 4];
    let _it = v.rchunks_exact_mut(0);
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
fn test_in_place_iterator_specialization() {
    let src: Box<[usize]> = box [1, 2, 3];
    let src_ptr = src.as_ptr();
    let sink: Box<_> = src.into_vec().into_iter().map(std::convert::identity).collect();
    let sink_ptr = sink.as_ptr();
    assert_eq!(src_ptr, sink_ptr);
}

#[test]
fn test_box_slice_clone() {
    let data = vec![vec![0, 1], vec![0], vec![1]];
    let data2 = data.clone().into_boxed_slice().clone().to_vec();

    assert_eq!(data, data2);
}

#[test]
#[allow(unused_must_use)] // here, we care about the side effects of `.clone()`
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_box_slice_clone_panics() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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

            Canary { count: self.count.clone(), panics: self.panics }
        }
    }

    let drop_count = Arc::new(AtomicUsize::new(0));
    let canary = Canary { count: drop_count.clone(), panics: false };
    let panic = Canary { count: drop_count.clone(), panics: true };

    std::panic::catch_unwind(move || {
        // When xs is dropped, +5.
        let xs =
            vec![canary.clone(), canary.clone(), canary.clone(), panic, canary].into_boxed_slice();

        // When panic is cloned, +3.
        xs.clone();
    })
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
#[should_panic(expected = "source slice length (4) does not match destination slice length (5)")]
fn test_copy_from_slice_dst_longer() {
    let src = [0, 1, 2, 3];
    let mut dst = [0; 5];
    dst.copy_from_slice(&src);
}

#[test]
#[should_panic(expected = "source slice length (4) does not match destination slice length (3)")]
fn test_copy_from_slice_dst_shorter() {
    let src = [0, 1, 2, 3];
    let mut dst = [0; 3];
    dst.copy_from_slice(&src);
}

const MAX_LEN: usize = 80;

static DROP_COUNTS: [AtomicUsize; MAX_LEN] = [
    // FIXME(RFC 1109): AtomicUsize is not Copy.
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

static VERSIONS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Eq)]
struct DropCounter {
    x: u32,
    id: usize,
    version: Cell<usize>,
}

impl PartialEq for DropCounter {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for DropCounter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.version.set(self.version.get() + 1);
        other.version.set(other.version.get() + 1);
        VERSIONS.fetch_add(2, Relaxed);
        self.x.partial_cmp(&other.x)
    }
}

impl Ord for DropCounter {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        DROP_COUNTS[self.id].fetch_add(1, Relaxed);
        VERSIONS.fetch_sub(self.version.get(), Relaxed);
    }
}

macro_rules! test {
    ($input:ident, $func:ident) => {
        let len = $input.len();

        // Work out the total number of comparisons required to sort
        // this array...
        let mut count = 0usize;
        $input.to_owned().$func(|a, b| {
            count += 1;
            a.cmp(b)
        });

        // ... and then panic on each and every single one.
        for panic_countdown in 0..count {
            // Refresh the counters.
            VERSIONS.store(0, Relaxed);
            for i in 0..len {
                DROP_COUNTS[i].store(0, Relaxed);
            }

            let v = $input.to_owned();
            let _ = std::panic::catch_unwind(move || {
                let mut v = v;
                let mut panic_countdown = panic_countdown;
                v.$func(|a, b| {
                    if panic_countdown == 0 {
                        SILENCE_PANIC.with(|s| s.set(true));
                        panic!();
                    }
                    panic_countdown -= 1;
                    a.cmp(b)
                })
            });

            // Check that the number of things dropped is exactly
            // what we expect (i.e., the contents of `v`).
            for (i, c) in DROP_COUNTS.iter().enumerate().take(len) {
                let count = c.load(Relaxed);
                assert!(count == 1, "found drop count == {} for i == {}, len == {}", count, i, len);
            }

            // Check that the most recent versions of values were dropped.
            assert_eq!(VERSIONS.load(Relaxed), 0);
        }
    };
}

thread_local!(static SILENCE_PANIC: Cell<bool> = Cell::new(false));

#[test]
#[cfg_attr(target_os = "emscripten", ignore)] // no threads
fn panic_safe() {
    panic::update_hook(move |prev, info| {
        if !SILENCE_PANIC.with(|s| s.get()) {
            prev(info);
        }
    });

    let mut rng = thread_rng();

    // Miri is too slow (but still need to `chain` to make the types match)
    let lens = if cfg!(miri) { (1..10).chain(0..0) } else { (1..20).chain(70..MAX_LEN) };
    let moduli: &[u32] = if cfg!(miri) { &[5] } else { &[5, 20, 50] };

    for len in lens {
        for &modulus in moduli {
            for &has_runs in &[false, true] {
                let mut input = (0..len)
                    .map(|id| DropCounter {
                        x: rng.next_u32() % modulus,
                        id: id,
                        version: Cell::new(0),
                    })
                    .collect::<Vec<_>>();

                if has_runs {
                    for c in &mut input {
                        c.x = c.id as u32;
                    }

                    for _ in 0..5 {
                        let a = rng.gen::<usize>() % len;
                        let b = rng.gen::<usize>() % len;
                        if a < b {
                            input[a..b].reverse();
                        } else {
                            input.swap(a, b);
                        }
                    }
                }

                test!(input, sort_by);
                test!(input, sort_unstable_by);
            }
        }
    }

    // Set default panic hook again.
    drop(panic::take_hook());
}

#[test]
fn repeat_generic_slice() {
    assert_eq!([1, 2].repeat(2), vec![1, 2, 1, 2]);
    assert_eq!([1, 2, 3, 4].repeat(0), vec![]);
    assert_eq!([1, 2, 3, 4].repeat(1), vec![1, 2, 3, 4]);
    assert_eq!([1, 2, 3, 4].repeat(3), vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
}

#[test]
#[allow(unreachable_patterns)]
fn subslice_patterns() {
    // This test comprehensively checks the passing static and dynamic semantics
    // of subslice patterns `..`, `x @ ..`, `ref x @ ..`, and `ref mut @ ..`
    // in slice patterns `[$($pat), $(,)?]` .

    #[derive(PartialEq, Debug, Clone)]
    struct N(u8);

    macro_rules! n {
        ($($e:expr),* $(,)?) => {
            [$(N($e)),*]
        }
    }

    macro_rules! c {
        ($inp:expr, $typ:ty, $out:expr $(,)?) => {
            assert_eq!($out, identity::<$typ>($inp))
        };
    }

    macro_rules! m {
        ($e:expr, $p:pat => $b:expr) => {
            match $e {
                $p => $b,
                _ => panic!(),
            }
        };
    }

    // == Slices ==

    // Matching slices using `ref` patterns:
    let mut v = vec![N(0), N(1), N(2), N(3), N(4)];
    let mut vc = (0..=4).collect::<Vec<u8>>();

    let [..] = v[..]; // Always matches.
    m!(v[..], [N(0), ref sub @ .., N(4)] => c!(sub, &[N], n![1, 2, 3]));
    m!(v[..], [N(0), ref sub @ ..] => c!(sub, &[N], n![1, 2, 3, 4]));
    m!(v[..], [ref sub @ .., N(4)] => c!(sub, &[N], n![0, 1, 2, 3]));
    m!(v[..], [ref sub @ .., _, _, _, _, _] => c!(sub, &[N], &n![] as &[N]));
    m!(v[..], [_, _, _, _, _, ref sub @ ..] => c!(sub, &[N], &n![] as &[N]));
    m!(vc[..], [x, .., y] => c!((x, y), (u8, u8), (0, 4)));

    // Matching slices using `ref mut` patterns:
    let [..] = v[..]; // Always matches.
    m!(v[..], [N(0), ref mut sub @ .., N(4)] => c!(sub, &mut [N], n![1, 2, 3]));
    m!(v[..], [N(0), ref mut sub @ ..] => c!(sub, &mut [N], n![1, 2, 3, 4]));
    m!(v[..], [ref mut sub @ .., N(4)] => c!(sub, &mut [N], n![0, 1, 2, 3]));
    m!(v[..], [ref mut sub @ .., _, _, _, _, _] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(v[..], [_, _, _, _, _, ref mut sub @ ..] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(vc[..], [x, .., y] => c!((x, y), (u8, u8), (0, 4)));

    // Matching slices using default binding modes (&):
    let [..] = &v[..]; // Always matches.
    m!(&v[..], [N(0), sub @ .., N(4)] => c!(sub, &[N], n![1, 2, 3]));
    m!(&v[..], [N(0), sub @ ..] => c!(sub, &[N], n![1, 2, 3, 4]));
    m!(&v[..], [sub @ .., N(4)] => c!(sub, &[N], n![0, 1, 2, 3]));
    m!(&v[..], [sub @ .., _, _, _, _, _] => c!(sub, &[N], &n![] as &[N]));
    m!(&v[..], [_, _, _, _, _, sub @ ..] => c!(sub, &[N], &n![] as &[N]));
    m!(&vc[..], [x, .., y] => c!((x, y), (&u8, &u8), (&0, &4)));

    // Matching slices using default binding modes (&mut):
    let [..] = &mut v[..]; // Always matches.
    m!(&mut v[..], [N(0), sub @ .., N(4)] => c!(sub, &mut [N], n![1, 2, 3]));
    m!(&mut v[..], [N(0), sub @ ..] => c!(sub, &mut [N], n![1, 2, 3, 4]));
    m!(&mut v[..], [sub @ .., N(4)] => c!(sub, &mut [N], n![0, 1, 2, 3]));
    m!(&mut v[..], [sub @ .., _, _, _, _, _] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(&mut v[..], [_, _, _, _, _, sub @ ..] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(&mut vc[..], [x, .., y] => c!((x, y), (&mut u8, &mut u8), (&mut 0, &mut 4)));

    // == Arrays ==
    let mut v = n![0, 1, 2, 3, 4];
    let vc = [0, 1, 2, 3, 4];

    // Matching arrays by value:
    m!(v.clone(), [N(0), sub @ .., N(4)] => c!(sub, [N; 3], n![1, 2, 3]));
    m!(v.clone(), [N(0), sub @ ..] => c!(sub, [N; 4], n![1, 2, 3, 4]));
    m!(v.clone(), [sub @ .., N(4)] => c!(sub, [N; 4], n![0, 1, 2, 3]));
    m!(v.clone(), [sub @ .., _, _, _, _, _] => c!(sub, [N; 0], n![] as [N; 0]));
    m!(v.clone(), [_, _, _, _, _, sub @ ..] => c!(sub, [N; 0], n![] as [N; 0]));
    m!(v.clone(), [x, .., y] => c!((x, y), (N, N), (N(0), N(4))));
    m!(v.clone(), [..] => ());

    // Matching arrays by ref patterns:
    m!(v, [N(0), ref sub @ .., N(4)] => c!(sub, &[N; 3], &n![1, 2, 3]));
    m!(v, [N(0), ref sub @ ..] => c!(sub, &[N; 4], &n![1, 2, 3, 4]));
    m!(v, [ref sub @ .., N(4)] => c!(sub, &[N; 4], &n![0, 1, 2, 3]));
    m!(v, [ref sub @ .., _, _, _, _, _] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(v, [_, _, _, _, _, ref sub @ ..] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(vc, [x, .., y] => c!((x, y), (u8, u8), (0, 4)));

    // Matching arrays by ref mut patterns:
    m!(v, [N(0), ref mut sub @ .., N(4)] => c!(sub, &mut [N; 3], &mut n![1, 2, 3]));
    m!(v, [N(0), ref mut sub @ ..] => c!(sub, &mut [N; 4], &mut n![1, 2, 3, 4]));
    m!(v, [ref mut sub @ .., N(4)] => c!(sub, &mut [N; 4], &mut n![0, 1, 2, 3]));
    m!(v, [ref mut sub @ .., _, _, _, _, _] => c!(sub, &mut [N; 0], &mut n![] as &mut [N; 0]));
    m!(v, [_, _, _, _, _, ref mut sub @ ..] => c!(sub, &mut [N; 0], &mut n![] as &mut [N; 0]));

    // Matching arrays by default binding modes (&):
    m!(&v, [N(0), sub @ .., N(4)] => c!(sub, &[N; 3], &n![1, 2, 3]));
    m!(&v, [N(0), sub @ ..] => c!(sub, &[N; 4], &n![1, 2, 3, 4]));
    m!(&v, [sub @ .., N(4)] => c!(sub, &[N; 4], &n![0, 1, 2, 3]));
    m!(&v, [sub @ .., _, _, _, _, _] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(&v, [_, _, _, _, _, sub @ ..] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(&v, [..] => ());
    m!(&v, [x, .., y] => c!((x, y), (&N, &N), (&N(0), &N(4))));

    // Matching arrays by default binding modes (&mut):
    m!(&mut v, [N(0), sub @ .., N(4)] => c!(sub, &mut [N; 3], &mut n![1, 2, 3]));
    m!(&mut v, [N(0), sub @ ..] => c!(sub, &mut [N; 4], &mut n![1, 2, 3, 4]));
    m!(&mut v, [sub @ .., N(4)] => c!(sub, &mut [N; 4], &mut n![0, 1, 2, 3]));
    m!(&mut v, [sub @ .., _, _, _, _, _] => c!(sub, &mut [N; 0], &mut n![] as &[N; 0]));
    m!(&mut v, [_, _, _, _, _, sub @ ..] => c!(sub, &mut [N; 0], &mut n![] as &[N; 0]));
    m!(&mut v, [..] => ());
    m!(&mut v, [x, .., y] => c!((x, y), (&mut N, &mut N), (&mut N(0), &mut N(4))));
}

#[test]
fn test_group_by() {
    let slice = &[1, 1, 1, 3, 3, 2, 2, 2, 1, 0];

    let mut iter = slice.group_by(|a, b| a == b);
    assert_eq!(iter.next(), Some(&[1, 1, 1][..]));
    assert_eq!(iter.next(), Some(&[3, 3][..]));
    assert_eq!(iter.next(), Some(&[2, 2, 2][..]));
    assert_eq!(iter.next(), Some(&[1][..]));
    assert_eq!(iter.next(), Some(&[0][..]));
    assert_eq!(iter.next(), None);

    let mut iter = slice.group_by(|a, b| a == b);
    assert_eq!(iter.next_back(), Some(&[0][..]));
    assert_eq!(iter.next_back(), Some(&[1][..]));
    assert_eq!(iter.next_back(), Some(&[2, 2, 2][..]));
    assert_eq!(iter.next_back(), Some(&[3, 3][..]));
    assert_eq!(iter.next_back(), Some(&[1, 1, 1][..]));
    assert_eq!(iter.next_back(), None);

    let mut iter = slice.group_by(|a, b| a == b);
    assert_eq!(iter.next(), Some(&[1, 1, 1][..]));
    assert_eq!(iter.next_back(), Some(&[0][..]));
    assert_eq!(iter.next(), Some(&[3, 3][..]));
    assert_eq!(iter.next_back(), Some(&[1][..]));
    assert_eq!(iter.next(), Some(&[2, 2, 2][..]));
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_group_by_mut() {
    let slice = &mut [1, 1, 1, 3, 3, 2, 2, 2, 1, 0];

    let mut iter = slice.group_by_mut(|a, b| a == b);
    assert_eq!(iter.next(), Some(&mut [1, 1, 1][..]));
    assert_eq!(iter.next(), Some(&mut [3, 3][..]));
    assert_eq!(iter.next(), Some(&mut [2, 2, 2][..]));
    assert_eq!(iter.next(), Some(&mut [1][..]));
    assert_eq!(iter.next(), Some(&mut [0][..]));
    assert_eq!(iter.next(), None);

    let mut iter = slice.group_by_mut(|a, b| a == b);
    assert_eq!(iter.next_back(), Some(&mut [0][..]));
    assert_eq!(iter.next_back(), Some(&mut [1][..]));
    assert_eq!(iter.next_back(), Some(&mut [2, 2, 2][..]));
    assert_eq!(iter.next_back(), Some(&mut [3, 3][..]));
    assert_eq!(iter.next_back(), Some(&mut [1, 1, 1][..]));
    assert_eq!(iter.next_back(), None);

    let mut iter = slice.group_by_mut(|a, b| a == b);
    assert_eq!(iter.next(), Some(&mut [1, 1, 1][..]));
    assert_eq!(iter.next_back(), Some(&mut [0][..]));
    assert_eq!(iter.next(), Some(&mut [3, 3][..]));
    assert_eq!(iter.next_back(), Some(&mut [1][..]));
    assert_eq!(iter.next(), Some(&mut [2, 2, 2][..]));
    assert_eq!(iter.next_back(), None);
}

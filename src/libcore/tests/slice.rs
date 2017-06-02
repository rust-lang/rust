// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::Ordering::{Equal, Greater, Less};
use core::slice::heapsort;
use core::result::Result::{Ok, Err};
use rand::{Rng, XorShiftRng};

#[test]
fn test_binary_search() {
    let b = [1, 2, 4, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&6)) == Ok(3));
    assert!(b.binary_search_by(|v| v.cmp(&5)) == Err(3));
    let b = [1, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&6)) == Ok(3));
    assert!(b.binary_search_by(|v| v.cmp(&5)) == Err(3));
    let b = [1, 2, 4, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&8)) == Ok(4));
    assert!(b.binary_search_by(|v| v.cmp(&7)) == Err(4));
    let b = [1, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&8)) == Ok(5));
    let b = [1, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&7)) == Err(5));
    assert!(b.binary_search_by(|v| v.cmp(&0)) == Err(0));
    let b = [1, 2, 4, 5, 6, 8];
    assert!(b.binary_search_by(|v| v.cmp(&9)) == Err(6));
}

#[test]
fn test_iterator_nth() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().nth(i).unwrap(), &v[i]);
    }
    assert_eq!(v.iter().nth(v.len()), None);

    let mut iter = v.iter();
    assert_eq!(iter.nth(2).unwrap(), &v[2]);
    assert_eq!(iter.nth(1).unwrap(), &v[4]);
}

#[test]
fn test_iterator_last() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    assert_eq!(v.iter().last().unwrap(), &4);
    assert_eq!(v[..1].iter().last().unwrap(), &0);
}

#[test]
fn test_iterator_count() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    assert_eq!(v.iter().count(), 5);

    let mut iter2 = v.iter();
    iter2.next();
    iter2.next();
    assert_eq!(iter2.count(), 3);
}

#[test]
fn test_chunks_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.chunks(3);
    assert_eq!(c.count(), 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.chunks(2);
    assert_eq!(c2.count(), 3);

    let v3: &[i32] = &[];
    let c3 = v3.chunks(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    assert_eq!(c.nth(1).unwrap()[1], 3);
    assert_eq!(c.next().unwrap()[0], 4);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.chunks(3);
    assert_eq!(c2.nth(1).unwrap()[1], 4);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.chunks(2);
    assert_eq!(c.last().unwrap()[1], 5);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.chunks(2);
    assert_eq!(c2.last().unwrap()[0], 4);
}

#[test]
fn test_chunks_mut_count() {
    let mut v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_mut(3);
    assert_eq!(c.count(), 2);

    let mut v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_mut(2);
    assert_eq!(c2.count(), 3);

    let mut v3: &mut [i32] = &mut [];
    let c3 = v3.chunks_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_mut_nth() {
    let mut v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_mut(2);
    assert_eq!(c.nth(1).unwrap()[1], 3);
    assert_eq!(c.next().unwrap()[0], 4);

    let mut v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c2 = v2.chunks_mut(3);
    assert_eq!(c2.nth(1).unwrap()[1], 4);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_mut(2);
    assert_eq!(c.last().unwrap()[1], 5);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_mut(2);
    assert_eq!(c2.last().unwrap()[0], 4);
}

#[test]
fn test_windows_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.windows(3);
    assert_eq!(c.count(), 4);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.windows(6);
    assert_eq!(c2.count(), 0);

    let v3: &[i32] = &[];
    let c3 = v3.windows(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_windows_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.windows(2);
    assert_eq!(c.nth(2).unwrap()[1], 3);
    assert_eq!(c.next().unwrap()[0], 3);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.windows(4);
    assert_eq!(c2.nth(1).unwrap()[1], 2);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_windows_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.windows(2);
    assert_eq!(c.last().unwrap()[1], 5);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.windows(2);
    assert_eq!(c2.last().unwrap()[0], 3);
}

#[test]
fn get_range() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    assert_eq!(v.get(..), Some(&[0, 1, 2, 3, 4, 5][..]));
    assert_eq!(v.get(..2), Some(&[0, 1][..]));
    assert_eq!(v.get(2..), Some(&[2, 3, 4, 5][..]));
    assert_eq!(v.get(1..4), Some(&[1, 2, 3][..]));
    assert_eq!(v.get(7..), None);
    assert_eq!(v.get(7..10), None);
}

#[test]
fn get_mut_range() {
    let mut v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    assert_eq!(v.get_mut(..), Some(&mut [0, 1, 2, 3, 4, 5][..]));
    assert_eq!(v.get_mut(..2), Some(&mut [0, 1][..]));
    assert_eq!(v.get_mut(2..), Some(&mut [2, 3, 4, 5][..]));
    assert_eq!(v.get_mut(1..4), Some(&mut [1, 2, 3][..]));
    assert_eq!(v.get_mut(7..), None);
    assert_eq!(v.get_mut(7..10), None);
}

#[test]
fn get_unchecked_range() {
    unsafe {
        let v: &[i32] = &[0, 1, 2, 3, 4, 5];
        assert_eq!(v.get_unchecked(..), &[0, 1, 2, 3, 4, 5][..]);
        assert_eq!(v.get_unchecked(..2), &[0, 1][..]);
        assert_eq!(v.get_unchecked(2..), &[2, 3, 4, 5][..]);
        assert_eq!(v.get_unchecked(1..4), &[1, 2, 3][..]);
    }
}

#[test]
fn get_unchecked_mut_range() {
    unsafe {
        let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
        assert_eq!(v.get_unchecked_mut(..), &mut [0, 1, 2, 3, 4, 5][..]);
        assert_eq!(v.get_unchecked_mut(..2), &mut [0, 1][..]);
        assert_eq!(v.get_unchecked_mut(2..), &mut[2, 3, 4, 5][..]);
        assert_eq!(v.get_unchecked_mut(1..4), &mut [1, 2, 3][..]);
    }
}

#[test]
fn test_find_rfind() {
    let v = [0, 1, 2, 3, 4, 5];
    let mut iter = v.iter();
    let mut i = v.len();
    while let Some(&elt) = iter.rfind(|_| true) {
        i -= 1;
        assert_eq!(elt, v[i]);
    }
    assert_eq!(i, 0);
    assert_eq!(v.iter().rfind(|&&x| x <= 3), Some(&3));
}

#[test]
fn test_rotate() {
    const N: usize = 600;
    let a: &mut [_] = &mut [0; N];
    for i in 0..N {
        a[i] = i;
    }

    a.rotate(42);
    let k = N - 42;

    for i in 0..N {
        assert_eq!(a[(i+k)%N], i);
    }
}

#[test]
fn sort_unstable() {
    let mut v = [0; 600];
    let mut tmp = [0; 600];
    let mut rng = XorShiftRng::new_unseeded();

    for len in (2..25).chain(500..510) {
        let v = &mut v[0..len];
        let tmp = &mut tmp[0..len];

        for &modulus in &[5, 10, 100, 1000] {
            for _ in 0..100 {
                for i in 0..len {
                    v[i] = rng.gen::<i32>() % modulus;
                }

                // Sort in default order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable();
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Sort in ascending order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable_by(|a, b| a.cmp(b));
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Sort in descending order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable_by(|a, b| b.cmp(a));
                assert!(tmp.windows(2).all(|w| w[0] >= w[1]));

                // Test heapsort using `<` operator.
                tmp.copy_from_slice(v);
                heapsort(tmp, |a, b| a < b);
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Test heapsort using `>` operator.
                tmp.copy_from_slice(v);
                heapsort(tmp, |a, b| a > b);
                assert!(tmp.windows(2).all(|w| w[0] >= w[1]));
            }
        }
    }

    // Sort using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    for i in 0..v.len() {
        v[i] = i as i32;
    }
    v.sort_unstable_by(|_, _| *rng.choose(&[Less, Equal, Greater]).unwrap());
    v.sort_unstable();
    for i in 0..v.len() {
        assert_eq!(v[i], i as i32);
    }

    // Should not panic.
    [0i32; 0].sort_unstable();
    [(); 10].sort_unstable();
    [(); 100].sort_unstable();

    let mut v = [0xDEADBEEFu64];
    v.sort_unstable();
    assert!(v == [0xDEADBEEF]);
}

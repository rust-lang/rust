// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::result::Result::{Ok, Err};


#[test]
fn test_position() {
    let b = [1, 2, 3, 5, 5];
    assert!(b.iter().position(|&v| v == 9) == None);
    assert!(b.iter().position(|&v| v == 5) == Some(3));
    assert!(b.iter().position(|&v| v == 3) == Some(2));
    assert!(b.iter().position(|&v| v == 0) == None);
}

#[test]
fn test_rposition() {
    let b = [1, 2, 3, 5, 5];
    assert!(b.iter().rposition(|&v| v == 9) == None);
    assert!(b.iter().rposition(|&v| v == 5) == Some(4));
    assert!(b.iter().rposition(|&v| v == 3) == Some(2));
    assert!(b.iter().rposition(|&v| v == 0) == None);
}

#[test]
fn test_binary_search() {
    let b: [i32; 0] = [];
    assert_eq!(b.binary_search(&5), Err(0));

    let b = [4];
    assert_eq!(b.binary_search(&3), Err(0));
    assert_eq!(b.binary_search(&4), Ok(0));
    assert_eq!(b.binary_search(&5), Err(1));

    let b = [1, 2, 4, 6, 8, 9];
    assert_eq!(b.binary_search(&5), Err(3));
    assert_eq!(b.binary_search(&6), Ok(3));
    assert_eq!(b.binary_search(&7), Err(4));
    assert_eq!(b.binary_search(&8), Ok(4));

    let b = [1, 2, 4, 5, 6, 8];
    assert_eq!(b.binary_search(&9), Err(6));

    let b = [1, 2, 4, 6, 7, 8, 9];
    assert_eq!(b.binary_search(&6), Ok(3));
    assert_eq!(b.binary_search(&5), Err(3));
    assert_eq!(b.binary_search(&8), Ok(5));

    let b = [1, 2, 4, 5, 6, 8, 9];
    assert_eq!(b.binary_search(&7), Err(5));
    assert_eq!(b.binary_search(&0), Err(0));

    let b = [1, 3, 3, 3, 7];
    assert_eq!(b.binary_search(&0), Err(0));
    assert_eq!(b.binary_search(&1), Ok(0));
    assert_eq!(b.binary_search(&2), Err(1));
    assert!(match b.binary_search(&3) { Ok(1...3) => true, _ => false });
    assert!(match b.binary_search(&3) { Ok(1...3) => true, _ => false });
    assert_eq!(b.binary_search(&4), Err(4));
    assert_eq!(b.binary_search(&5), Err(4));
    assert_eq!(b.binary_search(&6), Err(4));
    assert_eq!(b.binary_search(&7), Ok(4));
    assert_eq!(b.binary_search(&8), Err(5));
}

#[test]
// Test implementation specific behavior when finding equivalent elements.
// It is ok to break this test but when you do a crater run is highly advisable.
fn test_binary_search_implementation_details() {
    let b = [1, 1, 2, 2, 3, 3, 3];
    assert_eq!(b.binary_search(&1), Ok(1));
    assert_eq!(b.binary_search(&2), Ok(3));
    assert_eq!(b.binary_search(&3), Ok(6));
    let b = [1, 1, 1, 1, 1, 3, 3, 3, 3];
    assert_eq!(b.binary_search(&1), Ok(4));
    assert_eq!(b.binary_search(&3), Ok(8));
    let b = [1, 1, 1, 1, 3, 3, 3, 3, 3];
    assert_eq!(b.binary_search(&1), Ok(3));
    assert_eq!(b.binary_search(&3), Ok(8));
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
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.chunks(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4]);
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
fn test_chunks_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1.chunks(2)
        .zip(v2.chunks(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();
    assert_eq!(res, vec![14, 22, 14]);
}

#[test]
fn test_chunks_mut_count() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_mut(3);
    assert_eq!(c.count(), 2);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_mut(2);
    assert_eq!(c2.count(), 3);

    let v3: &mut [i32] = &mut [];
    let c3 = v3.chunks_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_mut_nth() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_mut(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c2 = v2.chunks_mut(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_mut(2);
    assert_eq!(c.last().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_mut(2);
    assert_eq!(c2.last().unwrap(), &[4]);
}

#[test]
fn test_chunks_mut_zip() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    for (a, b) in v1.chunks_mut(2).zip(v2.chunks(2)) {
        let sum = b.iter().sum::<i32>();
        for v in a {
            *v += sum;
        }
    }
    assert_eq!(v1, [13, 14, 19, 20, 14]);
}

#[test]
fn test_exact_chunks_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.exact_chunks(3);
    assert_eq!(c.count(), 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.exact_chunks(2);
    assert_eq!(c2.count(), 2);

    let v3: &[i32] = &[];
    let c3 = v3.exact_chunks(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_exact_chunks_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.exact_chunks(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.exact_chunks(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4, 5]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_exact_chunks_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.exact_chunks(2);
    assert_eq!(c.last().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.exact_chunks(2);
    assert_eq!(c2.last().unwrap(), &[2, 3]);
}

#[test]
fn test_exact_chunks_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1.exact_chunks(2)
        .zip(v2.exact_chunks(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();
    assert_eq!(res, vec![14, 22]);
}

#[test]
fn test_exact_chunks_mut_count() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.exact_chunks_mut(3);
    assert_eq!(c.count(), 2);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.exact_chunks_mut(2);
    assert_eq!(c2.count(), 2);

    let v3: &mut [i32] = &mut [];
    let c3 = v3.exact_chunks_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_exact_chunks_mut_nth() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.exact_chunks_mut(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.exact_chunks_mut(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4, 5]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_exact_chunks_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.exact_chunks_mut(2);
    assert_eq!(c.last().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.exact_chunks_mut(2);
    assert_eq!(c2.last().unwrap(), &[2, 3]);
}

#[test]
fn test_exact_chunks_mut_zip() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    for (a, b) in v1.exact_chunks_mut(2).zip(v2.exact_chunks(2)) {
        let sum = b.iter().sum::<i32>();
        for v in a {
            *v += sum;
        }
    }
    assert_eq!(v1, [13, 14, 19, 20, 4]);
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
fn test_windows_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1.windows(2)
        .zip(v2.windows(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();

    assert_eq!(res, [14, 18, 22, 26]);
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
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
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
fn test_iter_folds() {
    let a = [1, 2, 3, 4, 5]; // len>4 so the unroll is used
    assert_eq!(a.iter().fold(0, |acc, &x| 2*acc + x), 57);
    assert_eq!(a.iter().rfold(0, |acc, &x| 2*acc + x), 129);
    let fold = |acc: i32, &x| acc.checked_mul(2)?.checked_add(x);
    assert_eq!(a.iter().try_fold(0, &fold), Some(57));
    assert_eq!(a.iter().try_rfold(0, &fold), Some(129));

    // short-circuiting try_fold, through other methods
    let a = [0, 1, 2, 3, 5, 5, 5, 7, 8, 9];
    let mut iter = a.iter();
    assert_eq!(iter.position(|&x| x == 3), Some(3));
    assert_eq!(iter.rfind(|&&x| x == 5), Some(&5));
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_rotate_left() {
    const N: usize = 600;
    let a: &mut [_] = &mut [0; N];
    for i in 0..N {
        a[i] = i;
    }

    a.rotate_left(42);
    let k = N - 42;

    for i in 0..N {
        assert_eq!(a[(i + k) % N], i);
    }
}

#[test]
fn test_rotate_right() {
    const N: usize = 600;
    let a: &mut [_] = &mut [0; N];
    for i in 0..N {
        a[i] = i;
    }

    a.rotate_right(42);

    for i in 0..N {
        assert_eq!(a[(i + 42) % N], i);
    }
}

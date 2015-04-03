// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering::{Equal, Greater, Less};
use std::collections::{BitSet, BitVec};

#[test]
fn test_bit_set_show() {
    let mut s = BitSet::new();
    s.insert(1);
    s.insert(10);
    s.insert(50);
    s.insert(2);
    assert_eq!("{1, 2, 10, 50}", format!("{:?}", s));
}

#[test]
fn test_bit_set_from_usizes() {
    let usizes = vec![0, 2, 2, 3];
    let a: BitSet = usizes.into_iter().collect();
    let mut b = BitSet::new();
    b.insert(0);
    b.insert(2);
    b.insert(3);
    assert_eq!(a, b);
}

#[test]
fn test_bit_set_iterator() {
    let usizes = vec![0, 2, 2, 3];
    let bit_vec: BitSet = usizes.into_iter().collect();

    let idxs: Vec<_> = bit_vec.iter().collect();
    assert_eq!(idxs, [0, 2, 3]);

    let long: BitSet = (0..10000).filter(|&n| n % 2 == 0).collect();
    let real: Vec<_> = (0..10000).step_by(2).collect();

    let idxs: Vec<_> = long.iter().collect();
    assert_eq!(idxs, real);
}

#[test]
fn test_bit_set_frombit_vec_init() {
    let bools = [true, false];
    let lengths = [10, 64, 100];
    for &b in &bools {
        for &l in &lengths {
            let bitset = BitSet::from_bit_vec(BitVec::from_elem(l, b));
            assert_eq!(bitset.contains(&1), b);
            assert_eq!(bitset.contains(&(l-1)), b);
            assert!(!bitset.contains(&l));
        }
    }
}

#[test]
fn test_bit_vec_masking() {
    let b = BitVec::from_elem(140, true);
    let mut bs = BitSet::from_bit_vec(b);
    assert!(bs.contains(&139));
    assert!(!bs.contains(&140));
    assert!(bs.insert(150));
    assert!(!bs.contains(&140));
    assert!(!bs.contains(&149));
    assert!(bs.contains(&150));
    assert!(!bs.contains(&151));
}

#[test]
fn test_bit_set_basic() {
    let mut b = BitSet::new();
    assert!(b.insert(3));
    assert!(!b.insert(3));
    assert!(b.contains(&3));
    assert!(b.insert(4));
    assert!(!b.insert(4));
    assert!(b.contains(&3));
    assert!(b.insert(400));
    assert!(!b.insert(400));
    assert!(b.contains(&400));
    assert_eq!(b.len(), 3);
}

#[test]
fn test_bit_set_intersection() {
    let mut a = BitSet::new();
    let mut b = BitSet::new();

    assert!(a.insert(11));
    assert!(a.insert(1));
    assert!(a.insert(3));
    assert!(a.insert(77));
    assert!(a.insert(103));
    assert!(a.insert(5));

    assert!(b.insert(2));
    assert!(b.insert(11));
    assert!(b.insert(77));
    assert!(b.insert(5));
    assert!(b.insert(3));

    let expected = [3, 5, 11, 77];
    let actual: Vec<_> = a.intersection(&b).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_bit_set_difference() {
    let mut a = BitSet::new();
    let mut b = BitSet::new();

    assert!(a.insert(1));
    assert!(a.insert(3));
    assert!(a.insert(5));
    assert!(a.insert(200));
    assert!(a.insert(500));

    assert!(b.insert(3));
    assert!(b.insert(200));

    let expected = [1, 5, 500];
    let actual: Vec<_> = a.difference(&b).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_bit_set_symmetric_difference() {
    let mut a = BitSet::new();
    let mut b = BitSet::new();

    assert!(a.insert(1));
    assert!(a.insert(3));
    assert!(a.insert(5));
    assert!(a.insert(9));
    assert!(a.insert(11));

    assert!(b.insert(3));
    assert!(b.insert(9));
    assert!(b.insert(14));
    assert!(b.insert(220));

    let expected = [1, 5, 11, 14, 220];
    let actual: Vec<_> = a.symmetric_difference(&b).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_bit_set_union() {
    let mut a = BitSet::new();
    let mut b = BitSet::new();
    assert!(a.insert(1));
    assert!(a.insert(3));
    assert!(a.insert(5));
    assert!(a.insert(9));
    assert!(a.insert(11));
    assert!(a.insert(160));
    assert!(a.insert(19));
    assert!(a.insert(24));
    assert!(a.insert(200));

    assert!(b.insert(1));
    assert!(b.insert(5));
    assert!(b.insert(9));
    assert!(b.insert(13));
    assert!(b.insert(19));

    let expected = [1, 3, 5, 9, 11, 13, 19, 24, 160, 200];
    let actual: Vec<_> = a.union(&b).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_bit_set_subset() {
    let mut set1 = BitSet::new();
    let mut set2 = BitSet::new();

    assert!(set1.is_subset(&set2)); //  {}  {}
    set2.insert(100);
    assert!(set1.is_subset(&set2)); //  {}  { 1 }
    set2.insert(200);
    assert!(set1.is_subset(&set2)); //  {}  { 1, 2 }
    set1.insert(200);
    assert!(set1.is_subset(&set2)); //  { 2 }  { 1, 2 }
    set1.insert(300);
    assert!(!set1.is_subset(&set2)); // { 2, 3 }  { 1, 2 }
    set2.insert(300);
    assert!(set1.is_subset(&set2)); // { 2, 3 }  { 1, 2, 3 }
    set2.insert(400);
    assert!(set1.is_subset(&set2)); // { 2, 3 }  { 1, 2, 3, 4 }
    set2.remove(&100);
    assert!(set1.is_subset(&set2)); // { 2, 3 }  { 2, 3, 4 }
    set2.remove(&300);
    assert!(!set1.is_subset(&set2)); // { 2, 3 }  { 2, 4 }
    set1.remove(&300);
    assert!(set1.is_subset(&set2)); // { 2 }  { 2, 4 }
}

#[test]
fn test_bit_set_is_disjoint() {
    let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01000000]));
    let c = BitSet::new();
    let d = BitSet::from_bit_vec(BitVec::from_bytes(&[0b00110000]));

    assert!(!a.is_disjoint(&d));
    assert!(!d.is_disjoint(&a));

    assert!(a.is_disjoint(&b));
    assert!(a.is_disjoint(&c));
    assert!(b.is_disjoint(&a));
    assert!(b.is_disjoint(&c));
    assert!(c.is_disjoint(&a));
    assert!(c.is_disjoint(&b));
}

#[test]
fn test_bit_set_union_with() {
    //a should grow to include larger elements
    let mut a = BitSet::new();
    a.insert(0);
    let mut b = BitSet::new();
    b.insert(5);
    let expected = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10000100]));
    a.union_with(&b);
    assert_eq!(a, expected);

    // Standard
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let mut b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01100010]));
    let c = a.clone();
    a.union_with(&b);
    b.union_with(&c);
    assert_eq!(a.len(), 4);
    assert_eq!(b.len(), 4);
}

#[test]
fn test_bit_set_intersect_with() {
    // Explicitly 0'ed bits
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let mut b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b00000000]));
    let c = a.clone();
    a.intersect_with(&b);
    b.intersect_with(&c);
    assert!(a.is_empty());
    assert!(b.is_empty());

    // Uninitialized bits should behave like 0's
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let mut b = BitSet::new();
    let c = a.clone();
    a.intersect_with(&b);
    b.intersect_with(&c);
    assert!(a.is_empty());
    assert!(b.is_empty());

    // Standard
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let mut b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01100010]));
    let c = a.clone();
    a.intersect_with(&b);
    b.intersect_with(&c);
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
}

#[test]
fn test_bit_set_difference_with() {
    // Explicitly 0'ed bits
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b00000000]));
    let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    a.difference_with(&b);
    assert!(a.is_empty());

    // Uninitialized bits should behave like 0's
    let mut a = BitSet::new();
    let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b11111111]));
    a.difference_with(&b);
    assert!(a.is_empty());

    // Standard
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let mut b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01100010]));
    let c = a.clone();
    a.difference_with(&b);
    b.difference_with(&c);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
}

#[test]
fn test_bit_set_symmetric_difference_with() {
    //a should grow to include larger elements
    let mut a = BitSet::new();
    a.insert(0);
    a.insert(1);
    let mut b = BitSet::new();
    b.insert(1);
    b.insert(5);
    let expected = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10000100]));
    a.symmetric_difference_with(&b);
    assert_eq!(a, expected);

    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let b = BitSet::new();
    let c = a.clone();
    a.symmetric_difference_with(&b);
    assert_eq!(a, c);

    // Standard
    let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b11100010]));
    let mut b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01101010]));
    let c = a.clone();
    a.symmetric_difference_with(&b);
    b.symmetric_difference_with(&c);
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
}

#[test]
fn test_bit_set_eq() {
    let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b00000000]));
    let c = BitSet::new();

    assert!(a == a);
    assert!(a != b);
    assert!(a != c);
    assert!(b == b);
    assert!(b == c);
    assert!(c == c);
}

#[test]
fn test_bit_set_cmp() {
    let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100010]));
    let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b00000000]));
    let c = BitSet::new();

    assert_eq!(a.cmp(&b), Greater);
    assert_eq!(a.cmp(&c), Greater);
    assert_eq!(b.cmp(&a), Less);
    assert_eq!(b.cmp(&c), Equal);
    assert_eq!(c.cmp(&a), Less);
    assert_eq!(c.cmp(&b), Equal);
}

#[test]
fn test_bit_vec_remove() {
    let mut a = BitSet::new();

    assert!(a.insert(1));
    assert!(a.remove(&1));

    assert!(a.insert(100));
    assert!(a.remove(&100));

    assert!(a.insert(1000));
    assert!(a.remove(&1000));
    a.shrink_to_fit();
}

#[test]
fn test_bit_vec_clone() {
    let mut a = BitSet::new();

    assert!(a.insert(1));
    assert!(a.insert(100));
    assert!(a.insert(1000));

    let mut b = a.clone();

    assert!(a == b);

    assert!(b.remove(&1));
    assert!(a.contains(&1));

    assert!(a.remove(&1000));
    assert!(b.contains(&1000));
}

mod bench {
    use std::collections::{BitSet, BitVec};
    use std::rand::{Rng, self};
    use std::u32;

    use test::{Bencher, black_box};

    const BENCH_BITS : usize = 1 << 14;

    fn rng() -> rand::IsaacRng {
        let seed: &[_] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        rand::SeedableRng::from_seed(seed)
    }

    #[bench]
    fn bench_bit_vecset_small(b: &mut Bencher) {
        let mut r = rng();
        let mut bit_vec = BitSet::new();
        b.iter(|| {
            for _ in 0..100 {
                bit_vec.insert((r.next_u32() as usize) % u32::BITS);
            }
            black_box(&bit_vec);
        });
    }

    #[bench]
    fn bench_bit_vecset_big(b: &mut Bencher) {
        let mut r = rng();
        let mut bit_vec = BitSet::new();
        b.iter(|| {
            for _ in 0..100 {
                bit_vec.insert((r.next_u32() as usize) % BENCH_BITS);
            }
            black_box(&bit_vec);
        });
    }

    #[bench]
    fn bench_bit_vecset_iter(b: &mut Bencher) {
        let bit_vec = BitSet::from_bit_vec(BitVec::from_fn(BENCH_BITS,
                                              |idx| {idx % 3 == 0}));
        b.iter(|| {
            let mut sum = 0;
            for idx in &bit_vec {
                sum += idx as usize;
            }
            sum
        })
    }
}

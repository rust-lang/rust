// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::BTreeMap;
use std::collections::Bound::{Excluded, Included, Unbounded, self};
use std::collections::btree_map::Entry::{Occupied, Vacant};
use std::iter::range_inclusive;
use std::rc::Rc;

#[test]
fn test_basic_large() {
    let mut map = BTreeMap::new();
    let size = 10000;
    assert_eq!(map.len(), 0);

    for i in 0..size {
        assert_eq!(map.insert(i, 10*i), None);
        assert_eq!(map.len(), i + 1);
    }

    for i in 0..size {
        assert_eq!(map.get(&i).unwrap(), &(i*10));
    }

    for i in size..size*2 {
        assert_eq!(map.get(&i), None);
    }

    for i in 0..size {
        assert_eq!(map.insert(i, 100*i), Some(10*i));
        assert_eq!(map.len(), size);
    }

    for i in 0..size {
        assert_eq!(map.get(&i).unwrap(), &(i*100));
    }

    for i in 0..size/2 {
        assert_eq!(map.remove(&(i*2)), Some(i*200));
        assert_eq!(map.len(), size - i - 1);
    }

    for i in 0..size/2 {
        assert_eq!(map.get(&(2*i)), None);
        assert_eq!(map.get(&(2*i+1)).unwrap(), &(i*200 + 100));
    }

    for i in 0..size/2 {
        assert_eq!(map.remove(&(2*i)), None);
        assert_eq!(map.remove(&(2*i+1)), Some(i*200 + 100));
        assert_eq!(map.len(), size/2 - i - 1);
    }
}

#[test]
fn test_basic_small() {
    let mut map = BTreeMap::new();
    assert_eq!(map.remove(&1), None);
    assert_eq!(map.get(&1), None);
    assert_eq!(map.insert(1, 1), None);
    assert_eq!(map.get(&1), Some(&1));
    assert_eq!(map.insert(1, 2), Some(1));
    assert_eq!(map.get(&1), Some(&2));
    assert_eq!(map.insert(2, 4), None);
    assert_eq!(map.get(&2), Some(&4));
    assert_eq!(map.remove(&1), Some(2));
    assert_eq!(map.remove(&2), Some(4));
    assert_eq!(map.remove(&1), None);
}

#[test]
fn test_query() {
    // NOTE: shouldn't need to test mutable variants because they're
    // generated from the same template and deviate only in mutability
    // annotations. However incorrect implementations *may* result from Node's
    // mut and non-mut impls drifting.

    let size = 10000;
    let gap = 4;
    let min = gap;
    let max = (size - 1) * gap;

    let mut map = BTreeMap::new();

    assert_eq!(map.min(), None);
    assert_eq!(map.max(), None);
    assert_eq!(map.get_lt(&0), None);
    assert_eq!(map.get_gt(&0), None);
    assert_eq!(map.get_le(&0), None);
    assert_eq!(map.get_ge(&0), None);

    // linear insertions make degenerate trees, but we shouldn't care since all
    // large trees are structurally similar for these tests.
    for i in 1 .. size {
        // times 10 to make gaps
        map.insert(i * gap, i * gap);
    }

    assert_eq!(map.min(), Some((&min, &min)));
    assert_eq!(map.max(), Some((&max, &max)));


    // less exists checks
    for i in min + 1 .. max + gap {
        // gap = 4
        // input: 5  6  7  8  9  10 11 12
        // <=     4  4  4  8  8  8  8  12
        // <      4  4  4  4  8  8  8  8
        let le = i / gap * gap;
        let lt = (i - 1) / gap * gap;
        assert_eq!(map.get_lt(&i), Some((&lt, &lt)));
        assert_eq!(map.get_le(&i), Some((&le, &le)));
    }

    // greater exists checks
    for i in min - gap .. max - 1 {
        // gap = 4
        // input: 4  5  6  7  8  9  10 11
        // >=     4  8  8  8  8  12 12 12  (same as < but +4)
        // >      8  8  8  8  12 12 12 12  (same as <= but +4)
        let ge = (i - 1) / gap * gap + gap;
        let gt = i / gap * gap +  gap;
        assert_eq!(map.get_gt(&i), Some((&gt, &gt)));
        assert_eq!(map.get_ge(&i), Some((&ge, &ge)));
    }

    // less doesn't exist checks
    for i in 0 .. min {
        assert_eq!(map.get_lt(&i), None);
        assert_eq!(map.get_le(&i), None);
    }

    // greater doesn't exist checks
    for i in max + 1 .. max + gap  {
        assert_eq!(map.get_gt(&i), None);
        assert_eq!(map.get_ge(&i), None);
    }

    // special cases:
    assert_eq!(map.get_lt(&min), None);
    assert_eq!(map.get_le(&min), Some((&min, &min)));
    assert_eq!(map.get_gt(&max), None);
    assert_eq!(map.get_ge(&max), Some((&max, &max)));
}

#[test]
fn test_iter() {
    let size = 10000;

    // Forwards
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test<T>(size: usize, mut iter: T) where T: Iterator<Item=(usize, usize)> {
        for i in 0..size {
            assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
            assert_eq!(iter.next().unwrap(), (i, i));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }
    test(size, map.iter().map(|(&k, &v)| (k, v)));
    test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
    test(size, map.into_iter());
}

#[test]
fn test_iter_rev() {
    let size = 10000;

    // Forwards
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test<T>(size: usize, mut iter: T) where T: Iterator<Item=(usize, usize)> {
        for i in 0..size {
            assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
            assert_eq!(iter.next().unwrap(), (size - i - 1, size - i - 1));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }
    test(size, map.iter().rev().map(|(&k, &v)| (k, v)));
    test(size, map.iter_mut().rev().map(|(&k, &mut v)| (k, v)));
    test(size, map.into_iter().rev());
}

#[test]
fn test_iter_mixed() {
    let size = 10000;

    // Forwards
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test<T>(size: usize, mut iter: T)
            where T: Iterator<Item=(usize, usize)> + DoubleEndedIterator {
        for i in 0..size / 4 {
            assert_eq!(iter.size_hint(), (size - i * 2, Some(size - i * 2)));
            assert_eq!(iter.next().unwrap(), (i, i));
            assert_eq!(iter.next_back().unwrap(), (size - i - 1, size - i - 1));
        }
        for i in size / 4..size * 3 / 4 {
            assert_eq!(iter.size_hint(), (size * 3 / 4 - i, Some(size * 3 / 4 - i)));
            assert_eq!(iter.next().unwrap(), (i, i));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }
    test(size, map.iter().map(|(&k, &v)| (k, v)));
    test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
    test(size, map.into_iter());
}

#[test]
fn test_range_small() {
    let size = 5;

    // Forwards
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    let mut j = 0;
    for ((&k, &v), i) in map.range(Included(&2), Unbounded).zip(2..size) {
        assert_eq!(k, i);
        assert_eq!(v, i);
        j += 1;
    }
    assert_eq!(j, size - 2);
}

#[test]
fn test_range_1000() {
    let size = 1000;
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test(map: &BTreeMap<u32, u32>, size: u32, min: Bound<&u32>, max: Bound<&u32>) {
        let mut kvs = map.range(min, max).map(|(&k, &v)| (k, v));
        let mut pairs = (0..size).map(|i| (i, i));

        for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
            assert_eq!(kv, pair);
        }
        assert_eq!(kvs.next(), None);
        assert_eq!(pairs.next(), None);
    }
    test(&map, size, Included(&0), Excluded(&size));
    test(&map, size, Unbounded, Excluded(&size));
    test(&map, size, Included(&0), Included(&(size - 1)));
    test(&map, size, Unbounded, Included(&(size - 1)));
    test(&map, size, Included(&0), Unbounded);
    test(&map, size, Unbounded, Unbounded);
}

#[test]
fn test_range() {
    let size = 200;
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    for i in 0..size {
        for j in i..size {
            let mut kvs = map.range(Included(&i), Included(&j)).map(|(&k, &v)| (k, v));
            let mut pairs = range_inclusive(i, j).map(|i| (i, i));

            for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                assert_eq!(kv, pair);
            }
            assert_eq!(kvs.next(), None);
            assert_eq!(pairs.next(), None);
        }
    }
}

#[test]
fn test_borrow() {
    // make sure these compile -- using the Borrow trait
    {
        let mut map = BTreeMap::new();
        map.insert("0".to_string(), 1);
        assert_eq!(map["0"], 1);
    }

    {
        let mut map = BTreeMap::new();
        map.insert(Box::new(0), 1);
        assert_eq!(map[&0], 1);
    }

    {
        let mut map = BTreeMap::new();
        map.insert(Box::new([0, 1]) as Box<[i32]>, 1);
        assert_eq!(map[&[0, 1][..]], 1);
    }

    {
        let mut map = BTreeMap::new();
        map.insert(Rc::new(0), 1);
        assert_eq!(map[&0], 1);
    }
}

#[test]
fn test_entry(){
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map: BTreeMap<_, _> = xs.iter().cloned().collect();

    // Existing key (insert)
    match map.entry(1) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            assert_eq!(view.get(), &10);
            assert_eq!(view.insert(100), 10);
        }
    }
    assert_eq!(map.get(&1).unwrap(), &100);
    assert_eq!(map.len(), 6);


    // Existing key (update)
    match map.entry(2) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            let v = view.get_mut();
            *v *= 10;
        }
    }
    assert_eq!(map.get(&2).unwrap(), &200);
    assert_eq!(map.len(), 6);

    // Existing key (take)
    match map.entry(3) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.remove(), 30);
        }
    }
    assert_eq!(map.get(&3), None);
    assert_eq!(map.len(), 5);


    // Inexistent key (insert)
    match map.entry(10) {
        Occupied(_) => unreachable!(),
        Vacant(view) => {
            assert_eq!(*view.insert(1000), 1000);
        }
    }
    assert_eq!(map.get(&10).unwrap(), &1000);
    assert_eq!(map.len(), 6);
}

#[test]
fn test_extend_ref() {
    let mut a = BTreeMap::new();
    a.insert(1, "one");
    let mut b = BTreeMap::new();
    b.insert(2, "two");
    b.insert(3, "three");

    a.extend(&b);

    assert_eq!(a.len(), 3);
    assert_eq!(a[&1], "one");
    assert_eq!(a[&2], "two");
    assert_eq!(a[&3], "three");
}

mod bench {
    use std::collections::BTreeMap;
    use std::__rand::{Rng, thread_rng};
    use std::collections::Bound::{Included, Excluded, Unbounded};

    use test::{Bencher, black_box};

    fn get<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<&'a V> {
        map.get(key)
    }

    fn get_lt<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.get_lt(key)
    }
    fn get_le<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.get_le(key)
    }
    fn get_gt<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.get_gt(key)
    }
    fn get_ge<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.get_ge(key)
    }

    fn get_lt_range<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.range(Unbounded, Excluded(key)).next_back()
    }
    fn get_le_range<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.range(Unbounded, Included(key)).next_back()
    }
    fn get_gt_range<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.range(Excluded(key), Unbounded).next()
    }
    fn get_ge_range<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Option<(&'a K, &'a V)> {
        map.range(Included(key), Unbounded).next()
    }


    map_insert_rand_bench!{insert_rand_100,    100,    BTreeMap}
    map_insert_rand_bench!{insert_rand_10000, 10_000, BTreeMap}

    map_insert_seq_bench!{insert_seq_100,    100,    BTreeMap}
    map_insert_seq_bench!{insert_seq_10000, 10_000, BTreeMap}



    map_find_rand_bench!{get_rand_100_eq,    100,    BTreeMap, get}
    map_find_rand_bench!{get_rand_10000_eq, 10_000, BTreeMap, get}

    map_find_seq_bench!{get_seq_100_eq,    100,    BTreeMap, get}
    map_find_seq_bench!{get_seq_10000_eq, 10_000, BTreeMap, get}




    map_find_rand_bench!{get_rand_100_lt,    100,    BTreeMap, get_lt}
    map_find_rand_bench!{get_rand_10000_lt, 10_000, BTreeMap, get_lt}

    map_find_seq_bench!{get_seq_100_lt,    100,    BTreeMap, get_lt}
    map_find_seq_bench!{get_seq_10000_lt, 10_000, BTreeMap, get_lt}



    map_find_rand_bench!{get_rand_100_le,    100,    BTreeMap, get_le}
    map_find_rand_bench!{get_rand_10000_le, 10_000, BTreeMap, get_le}

    map_find_seq_bench!{get_seq_100_le,    100,    BTreeMap, get_le}
    map_find_seq_bench!{get_seq_10000_le, 10_000, BTreeMap, get_le}



    map_find_rand_bench!{get_rand_100_gt,    100,    BTreeMap, get_gt}
    map_find_rand_bench!{get_rand_10000_gt, 10_000, BTreeMap, get_gt}

    map_find_seq_bench!{get_seq_100_gt,    100,    BTreeMap, get_gt}
    map_find_seq_bench!{get_seq_10000_gt, 10_000, BTreeMap, get_gt}



    map_find_rand_bench!{get_rand_100_ge,    100,    BTreeMap, get_ge}
    map_find_rand_bench!{get_rand_10000_ge, 10_000, BTreeMap, get_ge}

    map_find_seq_bench!{get_seq_100_ge,    100,    BTreeMap, get_ge}
    map_find_seq_bench!{get_seq_10000_ge, 10_000, BTreeMap, get_ge}




    map_find_rand_bench!{get_rand_100_lt_range,    100,    BTreeMap, get_lt_range}
    map_find_rand_bench!{get_rand_10000_lt_range, 10_000, BTreeMap, get_lt_range}

    map_find_seq_bench!{get_seq_100_lt_range,    100,    BTreeMap, get_lt_range}
    map_find_seq_bench!{get_seq_10000_lt_range, 10_000, BTreeMap, get_lt_range}



    map_find_rand_bench!{get_rand_100_le_range,    100,    BTreeMap, get_le_range}
    map_find_rand_bench!{get_rand_10000_le_range, 10_000, BTreeMap, get_le_range}

    map_find_seq_bench!{get_seq_100_le_range,    100,    BTreeMap, get_le_range}
    map_find_seq_bench!{get_seq_10000_le_range, 10_000, BTreeMap, get_le_range}



    map_find_rand_bench!{get_rand_100_gt_range,    100,    BTreeMap, get_gt_range}
    map_find_rand_bench!{get_rand_10000_gt_range, 10_000, BTreeMap, get_gt_range}

    map_find_seq_bench!{get_seq_100_gt_range,    100,    BTreeMap, get_gt_range}
    map_find_seq_bench!{get_seq_10000_gt_range, 10_000, BTreeMap, get_gt_range}



    map_find_rand_bench!{get_rand_100_ge_range,    100,    BTreeMap, get_ge_range}
    map_find_rand_bench!{get_rand_10000_ge_range, 10_000, BTreeMap, get_ge_range}

    map_find_seq_bench!{get_seq_100_ge_range,    100,    BTreeMap, get_ge_range}
    map_find_seq_bench!{get_seq_10000_ge_range, 10_000, BTreeMap, get_ge_range}


    fn bench_iter(b: &mut Bencher, size: i32) {
        let mut map = BTreeMap::<i32, i32>::new();
        let mut rng = thread_rng();

        for _ in 0..size {
            map.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            for entry in &map {
                black_box(entry);
            }
        });
    }

    fn bench_iter_with_queries_forward(b: &mut Bencher, size: i32) {
        let mut map = BTreeMap::<i32, i32>::new();
        let mut rng = thread_rng();

        for _ in 0..size {
            map.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            let entry = map.min().unwrap();
            black_box(entry);
            let mut min = entry.0;
            while let Some(entry) = map.get_gt(min) {
                black_box(entry);
                min = entry.0;
            }
        });
    }

    fn bench_iter_with_queries_backward(b: &mut Bencher, size: i32) {
        let mut map = BTreeMap::<i32, i32>::new();
        let mut rng = thread_rng();

        for _ in 0..size {
            map.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            let entry = map.max().unwrap();
            black_box(entry);
            let mut max = entry.0;
            while let Some(entry) = map.get_lt(max) {
                black_box(entry);
                max = entry.0;
            }
        });
    }

    #[bench]
    pub fn iter_20_plain(b: &mut Bencher) {
        bench_iter(b, 20);
    }

    #[bench]
    pub fn iter_1000_plain(b: &mut Bencher) {
        bench_iter(b, 1000);
    }

    #[bench]
    pub fn iter_100000_plain(b: &mut Bencher) {
        bench_iter(b, 100000);
    }

    #[bench]
    pub fn iter_20_with_queries_forward(b: &mut Bencher) {
        bench_iter_with_queries_forward(b, 20);
    }

    #[bench]
    pub fn iter_1000_with_queries_forward(b: &mut Bencher) {
        bench_iter_with_queries_forward(b, 1000);
    }

    #[bench]
    pub fn iter_100000_with_queries_forward(b: &mut Bencher) {
        bench_iter_with_queries_forward(b, 100000);
    }

    #[bench]
    pub fn iter_20_with_queries_backward(b: &mut Bencher) {
        bench_iter_with_queries_backward(b, 20);
    }

    #[bench]
    pub fn iter_1000_with_queries_backward(b: &mut Bencher) {
        bench_iter_with_queries_backward(b, 1000);
    }

    #[bench]
    pub fn iter_100000_with_queries_backward(b: &mut Bencher) {
        bench_iter_with_queries_backward(b, 100000);
    }
}
